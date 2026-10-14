#include "detail/plan_validation.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace tileinfer::detail {
namespace {
using json = nlohmann::json;
using Id = TensorId;

void require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

int index(const json& value, std::int64_t limit)
{
    require(value.is_number_integer(), "program index must be an integer");
    if (value.is_number_unsigned())
        require(value.get<std::uint64_t>() < static_cast<std::uint64_t>(limit),
            "program index outside range");
    const auto result = value.get<std::int64_t>();
    require(result >= 0 && result < limit, "program index outside range");
    return static_cast<int>(result);
}

struct Region
{
    Id tensor;
    int begin;
    int end;
};

struct CheckedTask
{
    TaskRecord record{};
    int wave;
    int worker;
    std::int64_t work;
    std::vector<Region> reads;
    std::vector<Region> writes;
};

int width(Id tensor, const PlanShape& s)
{
    switch (tensor) {
    case Id::k:
    case Id::v:
    case Id::K:
    case Id::V:
        return s.nkv * s.dh;
    case Id::gate:
    case Id::up:
    case Id::product:
        return s.f;
    default:
        return s.d;
    }
}

CheckedTask check_task(const json& task, const PlanShape& s, int workers, int waves, bool bf16)
{
    CheckedTask result{};
    result.wave = index(task.at("wave"), waves);
    result.worker = index(task.at("worker"), workers);
    const auto& op = task.at("operands");
    auto code = task.at("opcode").get<std::string>();
    require(
        code.ends_with(bf16 ? "_bf16" : "_f32"), "opcode precision differs from numeric policy");
    if (bf16)
        code.replace(code.size() - 5, 5, "_f32");
    json expected;
    json outputs = json::array();
    auto read = [&](Id id, int begin, int end) { result.reads.push_back({id, begin, end}); };
    auto write = [&](Id id, int begin, int end) {
        result.writes.push_back({id, begin, end});
        outputs.push_back({{"tensor", tensor_index(id)}, {"rows", {begin, end}}});
    };
    auto rows = [&](int limit) {
        require(op.at("rows").is_array() && op.at("rows").size() == 2, "invalid rows");
        const int begin = index(op.at("rows")[0], limit);
        const int end = index(op.at("rows")[1], static_cast<std::int64_t>(limit) + 1);
        require(end > begin, "empty or reversed task rows");
        result.record.begin = begin;
        result.record.end = end;
        return std::pair{begin, end};
    };
    auto operands = [&](Id a, Id b, Id c) {
        result.record.a = tensor_index(a);
        result.record.b = tensor_index(b);
        result.record.c = tensor_index(c);
    };

    if (code == "rms_norm_f32") {
        result.record.opcode = Opcode::rms_norm_f32;
        const bool first = op.at("input") == tensor_index(Id::x);
        const Id input = first ? Id::x : Id::r;
        const Id scale = first ? Id::attn_norm : Id::ffn_norm;
        const Id output = first ? Id::u : Id::z;
        expected = {{"input", tensor_index(input)}, {"scale", tensor_index(scale)},
            {"output", tensor_index(output)}};
        operands(input, scale, output);
        read(input, 0, s.d);
        write(output, 0, s.d);
        result.work = s.d;
    } else if (code == "gemv_f32") {
        result.record.opcode = Opcode::gemv_f32;
        struct Projection
        {
            Id input;
            Id weight;
            Id output;
            Id residual;
        };
        constexpr std::array projections{Projection{Id::u, Id::Wq, Id::q, Id::none},
            Projection{Id::u, Id::Wk, Id::k, Id::none}, Projection{Id::u, Id::Wv, Id::v, Id::none},
            Projection{Id::attention, Id::Wo, Id::r, Id::x},
            Projection{Id::z, Id::Wgate, Id::gate, Id::none},
            Projection{Id::z, Id::Wup, Id::up, Id::none},
            Projection{Id::product, Id::Wdown, Id::y, Id::r}};
        const auto found = std::find_if(projections.begin(), projections.end(),
            [&](auto p) { return op.at("weight") == tensor_index(p.weight); });
        require(found != projections.end(), "unsupported GEMV weight");
        const auto p = *found;
        const auto [begin, end] = rows(width(p.output, s));
        expected = {{"input", tensor_index(p.input)}, {"weight", tensor_index(p.weight)},
            {"output", tensor_index(p.output)}, {"rows", {begin, end}},
            {"addend", p.residual == Id::none ? json(nullptr) : json(tensor_index(p.residual))}};
        operands(p.input, p.weight, p.output);
        result.record.d = p.residual == Id::none ? kNoTensor : tensor_index(p.residual);
        read(p.input, 0, width(p.input, s));
        if (p.residual != Id::none)
            read(p.residual, begin, end);
        write(p.output, begin, end);
        result.work = static_cast<std::int64_t>(end - begin) * width(p.input, s);
    } else if (code == "rope_query_f32") {
        result.record.opcode = Opcode::rope_query_f32;
        const int head = index(op.at("head"), s.nq);
        expected = {
            {"input", tensor_index(Id::q)}, {"output", tensor_index(Id::q_rot)}, {"head", head}};
        result.record.a = tensor_index(Id::q);
        result.record.b = tensor_index(Id::q_rot);
        result.record.head = head;
        read(Id::q, head * s.dh, (head + 1) * s.dh);
        write(Id::q_rot, head * s.dh, (head + 1) * s.dh);
        result.work = s.dh;
    } else if (code == "rope_cache_f32") {
        result.record.opcode = Opcode::rope_cache_f32;
        const int head = index(op.at("kv_head"), s.nkv);
        expected = {{"key", tensor_index(Id::k)}, {"value", tensor_index(Id::v)},
            {"key_cache", tensor_index(Id::K)}, {"value_cache", tensor_index(Id::V)},
            {"kv_head", head}};
        operands(Id::k, Id::v, Id::K);
        result.record.d = tensor_index(Id::V);
        result.record.kv_head = head;
        read(Id::k, head * s.dh, (head + 1) * s.dh);
        read(Id::v, head * s.dh, (head + 1) * s.dh);
        for (Id cache : {Id::K, Id::V}) {
            result.writes.push_back({cache, head * s.dh, (head + 1) * s.dh});
            outputs.push_back(
                {{"tensor", tensor_index(cache)}, {"position", "current"}, {"head", head}});
        }
        result.work = 2LL * s.dh;
    } else if (code == "attention_prefix_f32") {
        result.record.opcode = Opcode::attention_prefix_f32;
        const int head = index(op.at("head"), s.nq);
        const int kv_head = head / (s.nq / s.nkv);
        expected = {{"query", tensor_index(Id::q_rot)}, {"key_cache", tensor_index(Id::K)},
            {"value_cache", tensor_index(Id::V)}, {"output", tensor_index(Id::attention)},
            {"head", head}, {"kv_head", kv_head}};
        operands(Id::q_rot, Id::K, Id::V);
        result.record.d = tensor_index(Id::attention);
        result.record.head = head;
        result.record.kv_head = kv_head;
        read(Id::q_rot, head * s.dh, (head + 1) * s.dh);
        read(Id::K, kv_head * s.dh, (kv_head + 1) * s.dh);
        read(Id::V, kv_head * s.dh, (kv_head + 1) * s.dh);
        write(Id::attention, head * s.dh, (head + 1) * s.dh);
        result.work = 2LL * s.context * s.dh;
    } else if (code == "silu_product_f32") {
        result.record.opcode = Opcode::silu_product_f32;
        const auto [begin, end] = rows(s.f);
        expected = {{"gate", tensor_index(Id::gate)}, {"up", tensor_index(Id::up)},
            {"output", tensor_index(Id::product)}, {"rows", {begin, end}}};
        operands(Id::gate, Id::up, Id::product);
        read(Id::gate, begin, end);
        read(Id::up, begin, end);
        write(Id::product, begin, end);
        result.work = end - begin;
    } else {
        throw std::runtime_error("unsupported decoder opcode");
    }
    if (bf16)
        result.record.opcode = static_cast<Opcode>(static_cast<unsigned>(result.record.opcode) + 6);
    for (const auto& [key, value] : expected.items())
        if (value.is_number_integer())
            require(op.at(key).is_number_integer(), "operand must be an integer");
    require(op == expected && task.at("output") == outputs,
        "invalid task operands or output ownership");
    return result;
}
} // namespace

void validate_program(const json& document, const PlanShape& s, int workers, LoadedPlan& result)
{
    const auto& input = document.at("tasks");
    const auto& waves = document.at("waves");
    require(input.is_array() && !input.empty() &&
                input.size() < static_cast<std::size_t>(std::numeric_limits<int>::max()),
        "invalid task count");
    require(
        waves.is_array() && !waves.empty() && waves.size() <= input.size(), "invalid wave count");
    const int count = static_cast<int>(input.size());
    result.wave_count = static_cast<int>(waves.size());
    require(
        waves.size() <= std::numeric_limits<std::uint32_t>::max() / static_cast<unsigned>(workers),
        "too many worker ranges");
    std::vector<CheckedTask> tasks(count);
    std::vector<bool> seen(count);
    for (const auto& task : input) {
        const int id = index(task.at("id"), count);
        require(!seen[id], "duplicate task ID");
        seen[id] = true;
        tasks[id] = check_task(
            task, s, workers, result.wave_count, result.weight_dtype == WeightDtype::bfloat16);
    }
    struct Writer
    {
        int begin;
        int end;
        int task;
    };
    std::array<std::vector<Writer>, kTensorCount> writers;
    for (int id = 0; id < count; ++id)
        for (const auto region : tasks[id].writes)
            writers[tensor_index(region.tensor)].push_back({region.begin, region.end, id});
    constexpr std::array produced{Id::u, Id::q, Id::k, Id::v, Id::q_rot, Id::K, Id::V,
        Id::attention, Id::r, Id::z, Id::gate, Id::up, Id::product, Id::y};
    for (Id tensor : produced) {
        auto& entries = writers[tensor_index(tensor)];
        std::sort(entries.begin(), entries.end(), [](auto a, auto b) { return a.begin < b.begin; });
        int end = 0;
        for (const auto writer : entries) {
            require(writer.begin == end, "missing or overlapping output rows");
            end = writer.end;
        }
        require(end == width(tensor, s), "incomplete output ownership");
    }
    // Grid barriers enforce data dependencies. Graph edges and cost metadata
    // belong to the Python planner; the executor checks actual read/write order.
    for (int id = 0; id < count; ++id) {
        const auto& task = tasks[id];
        for (const auto region : task.reads) {
            if (region.tensor == Id::x)
                continue;
            for (const auto writer : writers[tensor_index(region.tensor)]) {
                if (writer.begin >= region.end)
                    break;
                if (writer.end > region.begin)
                    require(tasks[writer.task].wave < task.wave, "read before producer wave");
            }
        }
    }
    std::fill(seen.begin(), seen.end(), false);
    for (int wave = 0; wave < result.wave_count; ++wave) {
        const auto& entry = waves[wave];
        require(index(entry.at("id"), result.wave_count) == wave &&
                    entry.at("workers").is_array() &&
                    entry.at("workers").size() == static_cast<std::size_t>(workers),
            "invalid wave workers");
        std::vector<std::int64_t> loads(workers);
        for (int worker = 0; worker < workers; ++worker) {
            const auto& lane = entry.at("workers")[worker];
            require(lane.is_array(), "invalid worker task list");
            const auto begin = result.tasks.size();
            for (const auto& item : lane) {
                const int id = index(item, count);
                require(!seen[id] && tasks[id].wave == wave && tasks[id].worker == worker,
                    "task is duplicated or assigned to a different wave/worker");
                seen[id] = true;
                require(tasks[id].work <= std::numeric_limits<std::int64_t>::max() - loads[worker],
                    "worker work estimate overflow");
                loads[worker] += tasks[id].work;
                result.tasks.push_back(tasks[id].record);
            }
            result.ranges.push_back({static_cast<std::uint32_t>(begin),
                static_cast<std::uint32_t>(result.tasks.size() - begin)});
        }
        result.worker_loads.push_back(std::move(loads));
    }
    require(std::all_of(seen.begin(), seen.end(), [](bool value) { return value; }),
        "task is absent from the schedule");
}
} // namespace tileinfer::detail
