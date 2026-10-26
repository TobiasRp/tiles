// Grid barriers against counter waits around attention, on a real decoder
// package and the worker's own operators (S4 Phase 0b gate).
//
//   counter_probe PACKAGE [POSITION...]
//
// The probe runs a BF16 stack package's validated instruction streams with a
// copy of the worker's dispatch loop. Each mode drops some of the barriers
// around attention; a task after a dropped barrier first waits until every
// plan predecessor it would otherwise have met at that barrier has finished.
// Consumers with the same predecessors share one counter: each predecessor
// adds one when it finishes, and the consumer waits for launches x producers,
// so counters never need a reset. (A first version polled one done flag per
// predecessor; Wo's 16 sequential acquire loads per task made it slower than
// the barriers.) Waits change only ordering, so every mode must reproduce the
// barrier mode's logits bit for bit, also with random delays before tasks.
// The barrier mode is the reference and must time like the decoder itself.
// K/V holds synthetic values (within 0.03% of a decoded prefix, Section 15).

#include "detail/cuda_check.hpp"
#include "detail/uploaded_plan.cuh"
#include "ops.cuh"
#include "plan.hpp"

#include <cooperative_groups.h>
#include <cuda/__event/timed_event.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda/algorithm>
#include <cuda/atomic>
#include <cuda/buffer>
#include <cuda/launch>
#include <cuda/memory_pool>
#include <cuda/stream>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace tileinfer;
using json = nlohmann::json;
using detail::check;
using Bf16 = __nv_bfloat16;

// One mode's instruction streams and dependency waits, on the device.
struct Schedule
{
    const TaskRecord* instructions;
    const WorkerTaskRange* streams;
    const std::int32_t* wait_counter;    // per instruction: counter to wait for, or -1
    const std::uint32_t* wait_producers; // per instruction: that counter's producers
    const std::uint32_t* signal_begin;   // per instruction: first entry in signals
    const std::uint32_t* signal_count;   // per instruction: counters the task adds to
    const std::int32_t* signals;         // counter IDs
    std::uint32_t* counters;             // this mode's counters, zero before its first launch
    std::uint32_t launch;                // this mode's launch number, from 1
    std::uint32_t generation;            // any launch's number, for the delays
    std::uint32_t jitter;                // nonzero: a random delay before every task
};

// The worker's launch state for a BF16 stack, as in worker_kernel.cu.
struct Buffers
{
    LayerLayout layout;
    StackConfig stack;
    ops::Shape shape;
    TokenId token;
    Bf16* input;
    Bf16* output;
    Bf16* scratch;
    Bf16* kv;
    const Bf16* weights;
    Bf16* attention_scores;
    float* partials;
    float* logits;

    __device__ Bf16* hidden(int index) const { return index % 2 == 0 ? input : output; }

    __device__ ops::Layer<Bf16> layer(int index) const
    {
        const Bf16* w = weights + stack.layer_weights_offset + index * stack.layer_weight_stride;
        Bf16* cache = kv + index * stack.layer_kv_stride;
        float* stats = partials + shape.attention_chunks * shape.hidden_size;
        float* ffn_partials = stats + shape.attention_chunks * shape.query_heads * 2;
        return {hidden(index), hidden(index + 1), scratch + layout.q, scratch + layout.k,
            scratch + layout.q_rot, scratch + layout.attention, scratch + layout.r,
            scratch + layout.product, cache + layout.key, cache + layout.value, attention_scores,
            partials, stats, ffn_partials, w + layout.attn_norm, w + layout.Wq, w + layout.Wk,
            w + layout.Wv, w + layout.Wo, w + layout.ffn_norm, w + layout.Wgate, w + layout.Wup,
            w + layout.Wdown, nullptr, nullptr};
    }
};

__device__ inline std::uint32_t mix(std::uint32_t value)
{
    value *= 0x9e3779b1u;
    value ^= value >> 15;
    value *= 0x85ebca77u;
    return value ^ (value >> 13);
}

__global__ __launch_bounds__(kThreads, 1) void probe_worker(
    __grid_constant__ const Schedule schedule, __grid_constant__ const Buffers buffers)
{
    __shared__ ops::SharedStorage shared;
    const auto& shape = buffers.shape;
    const auto& stack = buffers.stack;
    const WorkerTaskRange stream = schedule.streams[blockIdx.x];
    for (std::uint32_t index = stream.begin; index < stream.begin + stream.count; ++index) {
        const TaskRecord task = schedule.instructions[index];
        const int begin = static_cast<int>(task.begin);
        const int end = static_cast<int>(task.end);
        if (task.opcode == Opcode::barrier) {
            if (shape.position + 1 > begin)
                cooperative_groups::this_grid().sync();
            continue;
        }
        const int counter = schedule.wait_counter[index];
        if (threadIdx.x == 0) {
            if (schedule.jitter != 0)
                __nanosleep(mix(index ^ (schedule.generation << 16)) % 4096);
            if (counter >= 0) {
                cuda::atomic_ref<std::uint32_t, cuda::thread_scope_device> count(
                    schedule.counters[counter]);
                const std::uint32_t target = schedule.launch * schedule.wait_producers[index];
                while (count.load(cuda::memory_order_acquire) < target)
                    __nanosleep(32);
            }
        }
        if (counter >= 0 || schedule.jitter != 0)
            __syncthreads();
        switch (task.opcode) {
        case Opcode::qkv:
            ops::qkv(begin, end, buffers.layer(task.layer), shape, shared);
            break;
        case Opcode::attention_scores:
            ops::attention_scores(
                task.kv_head, task.chunk, buffers.layer(task.layer), shape, shared);
            break;
        case Opcode::attention_values:
            ops::attention_values(
                task.kv_head, task.chunk, buffers.layer(task.layer), shape, shared);
            break;
        case Opcode::output_projection:
            ops::output_projection(begin, end, buffers.layer(task.layer), shape, shared);
            break;
        case Opcode::ffn:
            ops::ffn(begin, end, task.chunk, buffers.layer(task.layer), shape, shared);
            break;
        case Opcode::ffn_reduce:
            ops::ffn_reduce(begin, end, buffers.layer(task.layer), shape);
            break;
        case Opcode::embedding:
            ops::embedding_lookup(buffers.token, begin, end, shape.hidden_size,
                buffers.weights + stack.embedding_offset, buffers.hidden(0));
            break;
        case Opcode::final_rms_norm:
            ops::normalize(buffers.hidden(stack.layers), buffers.weights + stack.final_norm_offset,
                buffers.hidden(stack.layers + 1), begin, end, shape.hidden_size,
                shape.numeric.rms_norm_eps, shared);
            break;
        case Opcode::logits:
            ops::logits(buffers.hidden(stack.layers + 1), buffers.weights + stack.lm_head_offset,
                buffers.logits, begin, end, shape.hidden_size, shared);
            break;
        default:
            break;
        }
        __syncthreads();
        if (schedule.signal_count[index] != 0 && threadIdx.x == 0) {
            __threadfence();
            for (std::uint32_t k = 0; k < schedule.signal_count[index]; ++k) {
                cuda::atomic_ref<std::uint32_t, cuda::thread_scope_device> count(
                    schedule.counters[schedule.signals[schedule.signal_begin[index] + k]]);
                count.fetch_add(1, cuda::memory_order_release);
            }
        }
    }
}

// Deterministic values in [-1, 1), as the diagnostic context's synthetic prefix.
__global__ void fill_synthetic(Bf16* data, std::size_t count)
{
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
        i += stride) {
        unsigned hash = static_cast<unsigned>(i) * 0x9e3779b1u;
        hash ^= hash >> 15;
        hash *= 0x85ebca77u;
        hash ^= hash >> 13;
        data[i] = static_cast<Bf16>(static_cast<float>(hash & 0xffffu) / 32768.0f - 1.0f);
    }
}

// Barriers a mode drops, named by the opcodes of the waves on either side.
// Dropping the barrier before Wo also drops the one before the values wave:
// a fused prefix skips that barrier and its values tasks do nothing, so Wo
// would otherwise wait only for them and race the scores tasks.
struct Mode
{
    const char* name;
    std::vector<std::pair<Opcode, Opcode>> dropped;
};

const std::vector<Mode> kModes{
    {"barriers", {}},
    {"qkv_attention", {{Opcode::qkv, Opcode::attention_scores}}},
    {"attention_wo", {{Opcode::attention_scores, Opcode::attention_values},
                         {Opcode::attention_values, Opcode::output_projection}}},
    {"all", {{Opcode::qkv, Opcode::attention_scores},
                {Opcode::attention_scores, Opcode::attention_values},
                {Opcode::attention_values, Opcode::output_projection}}},
};

// One mode's host-side schedule, built from the validated streams and the
// plan's task waves and predecessors.
struct HostSchedule
{
    std::vector<TaskRecord> instructions;
    std::vector<WorkerTaskRange> streams;
    std::vector<std::int32_t> wait_counter;
    std::vector<std::uint32_t> wait_producers;
    std::vector<std::uint32_t> signal_begin;
    std::vector<std::uint32_t> signal_count;
    std::vector<std::int32_t> signals;
    int counters = 0;
    int dropped_barriers = 0;
    int most_producers = 0;
};

HostSchedule build(const LoadedPlan& plan, const json& document, const Mode& mode)
{
    const auto& tasks = document.at("tasks");
    std::vector<int> wave(tasks.size());
    std::vector<std::vector<int>> predecessors(tasks.size());
    for (const auto& task : tasks) {
        const int id = task.at("id").get<int>();
        wave[id] = task.at("wave").get<int>();
        predecessors[id] = task.at("predecessors").get<std::vector<int>>();
    }
    const int waves = static_cast<int>(document.at("waves").size());
    // The opcode every task of a wave shares; the plan never mixes them.
    std::vector<Opcode> wave_opcode(waves, Opcode::barrier);
    for (std::size_t index = 0; index < plan.instructions.size(); ++index)
        if (plan.instruction_tasks[index] >= 0)
            wave_opcode[wave[plan.instruction_tasks[index]]] = plan.instructions[index].opcode;
    std::vector<bool> dropped(waves, false); // the barrier after wave w
    for (int w = 0; w + 1 < waves; ++w)
        for (const auto& [before, after] : mode.dropped)
            dropped[w] = dropped[w] || (wave_opcode[w] == before && wave_opcode[w + 1] == after);

    HostSchedule result;
    std::map<std::vector<int>, int> counter_of; // predecessor set -> counter
    std::vector<int> instruction_tasks;
    for (const auto& range : plan.streams) {
        const auto first = static_cast<std::uint32_t>(result.instructions.size());
        int barrier = 0;
        for (std::uint32_t index = range.begin; index < range.begin + range.count; ++index) {
            const int task = plan.instruction_tasks[index];
            if (task < 0) {
                if (dropped[barrier++]) {
                    ++result.dropped_barriers;
                    continue;
                }
                result.wait_counter.push_back(-1);
                result.wait_producers.push_back(0);
            } else {
                // Predecessors no kept barrier separates from this task. Every
                // one is in an earlier wave, so it is earlier in the global
                // order every worker's stream follows: waits cannot deadlock.
                std::vector<int> set;
                for (const int p : predecessors[task]) {
                    if (wave[p] >= wave[task])
                        throw std::runtime_error("predecessor in a later wave");
                    bool separated = false;
                    for (int w = wave[p]; w < wave[task]; ++w)
                        separated = separated || !dropped[w];
                    if (!separated)
                        set.push_back(p);
                }
                if (set.empty()) {
                    result.wait_counter.push_back(-1);
                } else {
                    std::ranges::sort(set);
                    const auto [entry, added] =
                        counter_of.try_emplace(set, static_cast<int>(counter_of.size()));
                    result.wait_counter.push_back(entry->second);
                }
                result.wait_producers.push_back(static_cast<std::uint32_t>(set.size()));
                result.most_producers =
                    std::max(result.most_producers, static_cast<int>(set.size()));
            }
            result.instructions.push_back(plan.instructions[index]);
            instruction_tasks.push_back(task);
        }
        result.streams.push_back(
            {first, static_cast<std::uint32_t>(result.instructions.size()) - first});
    }
    // Every counter whose predecessor set holds a task gets one add from it.
    std::map<int, std::vector<int>> counters_of_task;
    for (const auto& [set, counter] : counter_of)
        for (const int task : set)
            counters_of_task[task].push_back(counter);
    for (const int task : instruction_tasks) {
        result.signal_begin.push_back(static_cast<std::uint32_t>(result.signals.size()));
        const auto found = counters_of_task.find(task);
        const std::size_t count = found == counters_of_task.end() ? 0 : found->second.size();
        if (count != 0)
            result.signals.insert(result.signals.end(), found->second.begin(), found->second.end());
        result.signal_count.push_back(static_cast<std::uint32_t>(count));
    }
    result.counters = static_cast<int>(counter_of.size());
    result.dropped_barriers /= static_cast<int>(plan.streams.size());
    return result;
}

double elapsed_us(const cuda::timed_event& start, const cuda::timed_event& end)
{
    return cuda::std::chrono::duration<double, cuda::std::micro>(end - start).count();
}

template <typename T>
cuda::device_buffer<T> upload(
    cuda::stream& stream, cuda::device_memory_pool_ref& pool, const std::vector<T>& values)
{
    cuda::device_buffer<T> buffer(
        stream, pool, std::max<std::size_t>(values.size(), 1), cuda::no_init);
    if (!values.empty())
        cuda::copy_bytes(stream, values, buffer);
    return buffer;
}

struct DeviceSchedule
{
    HostSchedule host;
    cuda::device_buffer<TaskRecord> instructions;
    cuda::device_buffer<WorkerTaskRange> streams;
    cuda::device_buffer<std::int32_t> wait_counter;
    cuda::device_buffer<std::uint32_t> wait_producers;
    cuda::device_buffer<std::uint32_t> signal_begin;
    cuda::device_buffer<std::uint32_t> signal_count;
    cuda::device_buffer<std::int32_t> signals;
    cuda::device_buffer<std::uint32_t> counters;
    std::uint32_t launches = 0;

    DeviceSchedule(cuda::stream& stream, cuda::device_memory_pool_ref& pool, HostSchedule schedule)
        : host(std::move(schedule))
        , instructions(upload(stream, pool, host.instructions))
        , streams(upload(stream, pool, host.streams))
        , wait_counter(upload(stream, pool, host.wait_counter))
        , wait_producers(upload(stream, pool, host.wait_producers))
        , signal_begin(upload(stream, pool, host.signal_begin))
        , signal_count(upload(stream, pool, host.signal_count))
        , signals(upload(stream, pool, host.signals))
        , counters(upload(
              stream, pool, std::vector<std::uint32_t>(static_cast<std::size_t>(host.counters), 0)))
    {
    }
};

json run(const std::string& package, std::vector<int> positions)
{
    const auto target = describe_target();
    const LoadedPlan plan = load_plan(package, target, false);
    if (plan.weight_dtype != WeightDtype::bfloat16 || !plan.stack.full_stack)
        throw std::runtime_error("counter_probe needs a BF16 stack package");
    std::ifstream source(std::filesystem::path(package) / "plan.json");
    const json document = json::parse(source);

    const detail::UploadedPlan model(plan);
    const auto view = model.view();
    cuda::stream stream{model.device};
    auto& pool = model.pool;
    auto bf16 = [&](std::size_t count) {
        return cuda::device_buffer<Bf16>(
            stream, pool, std::max<std::size_t>(count, 1), cuda::no_init);
    };
    auto input = bf16(plan.hidden_size);
    auto output = bf16(plan.hidden_size);
    auto scratch = bf16(plan.scratch_bytes / 2);
    auto kv = bf16(plan.kv_bytes / 2);
    auto scores = bf16(plan.attention_bytes / 2);
    cuda::device_buffer<float> partials(
        stream, pool, (plan.attention_partial_bytes + plan.ffn_partial_bytes) / 4, cuda::no_init);
    cuda::device_buffer<float> logits(stream, pool, plan.stack.vocab_size, cuda::no_init);
    cuda::launch(stream, cuda::make_config(cuda::block_dims<256>(), cuda::grid_dims(1024)),
        fill_synthetic, kv.data(), kv.size());

    std::vector<DeviceSchedule> schedules;
    for (const auto& mode : kModes)
        schedules.emplace_back(stream, pool, build(plan, document, mode));
    std::uint32_t generation = 0;

    Buffers buffers{view.layout, view.stack,
        {view.hidden_size, view.intermediate_size, view.query_heads, view.kv_heads, view.head_dim,
            0, view.max_context, view.attention_chunks, view.attention_chunk_steps,
            view.attention_fused_steps, view.ffn_chunks, view.numeric, view.rotary_frequencies},
        1, input.data(), output.data(), scratch.data(), kv.data(),
        static_cast<const Bf16*>(view.weights), scores.data(), partials.data(), logits.data()};
    const auto config = cuda::make_config(
        cuda::block_dims<kThreads>(), cuda::grid_dims(view.workers), cuda::cooperative_launch{});
    auto launch = [&](DeviceSchedule& schedule, std::uint32_t jitter) {
        const Schedule device{schedule.instructions.data(), schedule.streams.data(),
            schedule.wait_counter.data(), schedule.wait_producers.data(),
            schedule.signal_begin.data(), schedule.signal_count.data(), schedule.signals.data(),
            schedule.counters.data(), ++schedule.launches, ++generation, jitter};
        cuda::launch(stream, config, probe_worker, device, buffers);
    };
    auto read_logits = [&] {
        std::vector<float> host(plan.stack.vocab_size);
        cuda::copy_bytes(stream, logits, host);
        stream.sync();
        return host;
    };

    json modes = json::array();
    for (const auto& schedule : schedules)
        modes.push_back({{"dropped_barriers_per_worker", schedule.host.dropped_barriers},
            {"counters", schedule.host.counters},
            {"most_producers_per_counter", schedule.host.most_producers}});
    for (std::size_t m = 0; m < kModes.size(); ++m)
        modes[m]["mode"] = kModes[m].name;

    constexpr int kWarmup = 3, kLaunches = 20, kRounds = 7, kJitterRuns = 3;
    json runs = json::array();
    for (const int position : positions) {
        if (position < 0 || position >= plan.max_context)
            throw std::runtime_error("position outside the package context");
        buffers.shape.position = position;
        // Bit identity: every mode, with and without delays, against barriers.
        launch(schedules[0], 0);
        const auto reference = read_logits();
        json identical = json::object();
        for (std::size_t m = 1; m < schedules.size(); ++m) {
            bool same = true;
            for (int run = 0; run <= kJitterRuns; ++run) {
                launch(schedules[m], run == 0 ? 0 : 1);
                const auto values = read_logits();
                same = same && std::memcmp(values.data(), reference.data(),
                                   values.size() * sizeof(float)) == 0;
            }
            identical[kModes[m].name] = same;
        }
        for (auto& schedule : schedules)
            for (int i = 0; i < kWarmup; ++i)
                launch(schedule, 0);
        stream.sync();
        std::vector<std::vector<double>> samples(schedules.size());
        for (int round = 0; round < kRounds; ++round)
            for (std::size_t n = 0; n < schedules.size(); ++n) {
                const std::size_t m = round % 2 == 0 ? n : schedules.size() - 1 - n;
                cuda::timed_event start{stream};
                start.record(stream);
                for (int i = 0; i < kLaunches; ++i)
                    launch(schedules[m], 0);
                cuda::timed_event end{stream};
                end.record(stream);
                end.sync();
                samples[m].push_back(elapsed_us(start, end) / kLaunches);
            }
        auto median = [](std::vector<double> values) {
            std::ranges::sort(values);
            return values[values.size() / 2];
        };
        json result{{"position", position}, {"bit_identical", identical}};
        for (std::size_t m = 0; m < schedules.size(); ++m) {
            std::vector<double> saving;
            for (int round = 0; round < kRounds; ++round)
                saving.push_back(samples[0][round] - samples[m][round]);
            result["modes"][kModes[m].name] = {
                {"us", median(samples[m])}, {"saving_us", median(saving)}, {"samples", samples[m]}};
        }
        runs.push_back(result);
    }
    return {{"package", package}, {"plan_sha256", plan.plan_sha256}, {"modes", modes},
        {"launches_per_sample", kLaunches}, {"rounds", kRounds}, {"runs", runs}};
}
} // namespace

int main(int argc, char** argv)
{
    try {
        if (argc < 2)
            throw std::runtime_error("usage: counter_probe PACKAGE [POSITION...]");
        std::vector<int> positions;
        for (int i = 2; i < argc; ++i)
            positions.push_back(std::stoi(argv[i]));
        if (positions.empty())
            positions = {0, 255, 383, 384, 1023, 2047, 4095};
        std::cout << run(argv[1], positions).dump() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "counter_probe: " << error.what() << '\n';
        return 1;
    }
}
