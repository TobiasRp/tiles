#include "detail/plan_validation.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
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

// Stack stages own regions outside any layer's tensors.
enum class Stage
{
    embedding,
    final_norm,
    logits,
};
constexpr int kStageCount = 3;

// Physical storage the worker reads or writes while executing one token.
// Scratch is shared by all layers, the two hidden buffers alternate, and each
// layer appends its own K/V row.
struct Buffers
{
    static constexpr int hidden = 0;  // hidden + (h % 2)
    static constexpr int scratch = 2; // scratch + tensor index
    static constexpr int scores = scratch + static_cast<int>(kTensorCount);
    static constexpr int logits = scores + 1;
    static constexpr int kv = logits + 1; // kv + 2 * layer + (0 for K, 1 for V)
};

// One contiguous element range and the logical value it reads or writes.
// Values are numbered per buffer: hidden state h_k is k, the final normalized
// hidden state is layers + 1, and every other write in layer l is l + 1.
struct Access
{
    int buffer;
    int begin;
    int end;
    int version;
};

// A logical output region used to check complete, nonoverlapping ownership.
struct Region
{
    int key; // layer * kTensorCount + tensor, or stage key after all layers
    int begin;
    int end;
};

struct CheckedTask
{
    TaskRecord record{};
    int wave;
    int worker;
    std::int64_t work;
    std::vector<Access> reads;
    std::vector<Access> writes;
    std::vector<Region> owned;
};

struct Program
{
    const PlanShape& s;
    int layers;
    int vocab;
    bool full_stack;
    bool bf16;

    [[nodiscard]] int width(Id tensor) const
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

    [[nodiscard]] int stage_key(Stage stage) const
    {
        return layers * static_cast<int>(kTensorCount) + static_cast<int>(stage);
    }

    [[nodiscard]] int stage_width(Stage stage) const
    {
        return stage == Stage::logits ? vocab : s.d;
    }

    CheckedTask check(const json& task, int workers, int waves) const;
};

CheckedTask Program::check(const json& task, int workers, int waves) const
{
    CheckedTask result{};
    result.wave = index(task.at("wave"), waves);
    result.worker = index(task.at("worker"), workers);
    const auto& op = task.at("operands");
    auto code = task.at("opcode").get<std::string>();
    const std::string suffix = bf16 ? "_bf16" : "_f32";
    require(code.ends_with(suffix), "opcode precision differs from numeric policy");
    code.resize(code.size() - suffix.size());

    const bool stage = code == "embedding" || code == "final_rms_norm" || code == "logits";
    int layer = 0;
    if (stage) {
        require(full_stack, "stack stage in a block package");
        require(task.at("layer").is_null(), "stack stage tagged with a layer");
        result.record.layer = kNoLayer;
    } else {
        layer = index(task.at("layer"), layers);
        result.record.layer = static_cast<std::uint32_t>(layer);
    }
    // Versions of this layer's input hidden state, output hidden state, and writes.
    const int input_hidden = Buffers::hidden + layer % 2;
    const int output_hidden = Buffers::hidden + (layer + 1) % 2;
    const int produced = layer + 1;

    json expected;
    json outputs = json::array();
    auto buffer = [&](Id id) {
        if (id == Id::x)
            return input_hidden;
        if (id == Id::y)
            return output_hidden;
        if (id == Id::K || id == Id::V)
            return Buffers::kv + 2 * layer + (id == Id::V ? 1 : 0);
        return Buffers::scratch + static_cast<int>(tensor_index(id));
    };
    auto version = [&](Id id) {
        if (id == Id::x)
            return layer;
        return id == Id::K || id == Id::V ? 1 : produced;
    };
    auto read = [&](Id id, int begin, int end) {
        result.reads.push_back({buffer(id), begin, end, version(id)});
    };
    auto write = [&](Id id, int begin, int end, bool listed = true) {
        result.writes.push_back({buffer(id), begin, end, version(id)});
        result.owned.push_back(
            {layer * static_cast<int>(kTensorCount) + static_cast<int>(tensor_index(id)), begin,
                end});
        if (listed)
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
    auto stage_output = [&](Stage kind, const char* name, int begin, int end) {
        result.owned.push_back({stage_key(kind), begin, end});
        outputs.push_back({{"stage", name}, {"rows", {begin, end}}});
    };

    if (code == "embedding") {
        result.record.opcode = Opcode::embedding;
        const auto [begin, end] = rows(s.d);
        expected = {{"rows", {begin, end}}};
        result.writes.push_back({Buffers::hidden, begin, end, 0});
        stage_output(Stage::embedding, "embedding", begin, end);
        result.work = end - begin;
    } else if (code == "final_rms_norm") {
        result.record.opcode = Opcode::final_rms_norm;
        const auto [begin, end] = rows(s.d);
        expected = {{"rows", {begin, end}}};
        result.reads.push_back({Buffers::hidden + layers % 2, 0, s.d, layers});
        result.writes.push_back({Buffers::hidden + (layers + 1) % 2, begin, end, layers + 1});
        stage_output(Stage::final_norm, "final_norm", begin, end);
        result.work = s.d + end - begin;
    } else if (code == "logits") {
        result.record.opcode = Opcode::logits;
        const auto [begin, end] = rows(vocab);
        expected = {{"rows", {begin, end}}};
        result.reads.push_back({Buffers::hidden + (layers + 1) % 2, 0, s.d, layers + 1});
        result.writes.push_back({Buffers::logits, begin, end, 1});
        stage_output(Stage::logits, "logits", begin, end);
        result.work = static_cast<std::int64_t>(end - begin) * s.d;
    } else if (code == "rms_norm") {
        result.record.opcode = Opcode::rms_norm;
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
    } else if (code == "gemv") {
        result.record.opcode = Opcode::gemv;
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
        const auto [begin, end] = rows(width(p.output));
        expected = {{"input", tensor_index(p.input)}, {"weight", tensor_index(p.weight)},
            {"output", tensor_index(p.output)}, {"rows", {begin, end}},
            {"addend", p.residual == Id::none ? json(nullptr) : json(tensor_index(p.residual))}};
        operands(p.input, p.weight, p.output);
        result.record.d = p.residual == Id::none ? kNoTensor : tensor_index(p.residual);
        read(p.input, 0, width(p.input));
        if (p.residual != Id::none)
            read(p.residual, begin, end);
        write(p.output, begin, end);
        result.work = static_cast<std::int64_t>(end - begin) * width(p.input);
    } else if (code == "rope_query") {
        result.record.opcode = Opcode::rope_query;
        const int head = index(op.at("head"), s.nq);
        expected = {
            {"input", tensor_index(Id::q)}, {"output", tensor_index(Id::q_rot)}, {"head", head}};
        result.record.a = tensor_index(Id::q);
        result.record.b = tensor_index(Id::q_rot);
        result.record.head = head;
        read(Id::q, head * s.dh, (head + 1) * s.dh);
        write(Id::q_rot, head * s.dh, (head + 1) * s.dh);
        result.work = s.dh;
    } else if (code == "rope_cache") {
        result.record.opcode = Opcode::rope_cache;
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
            write(cache, head * s.dh, (head + 1) * s.dh, false);
            outputs.push_back(
                {{"tensor", tensor_index(cache)}, {"position", "current"}, {"head", head}});
        }
        result.work = 2LL * s.dh;
    } else if (code == "attention_prefix") {
        result.record.opcode = Opcode::attention_prefix;
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
        // The head's score workspace row is private to this task.
        result.writes.push_back({Buffers::scores, head, head + 1, produced});
        result.work = 2LL * s.context * s.dh;
    } else if (code == "silu_product") {
        result.record.opcode = Opcode::silu_product;
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
    for (const auto& [key, value] : expected.items())
        if (value.is_number_integer())
            require(op.at(key).is_number_integer(), "operand must be an integer");
    require(op == expected && task.at("output") == outputs,
        "invalid task operands or output ownership");
    return result;
}

// Requires every logical output to be written exactly once per token.
void check_ownership(const Program& program, const std::vector<CheckedTask>& tasks)
{
    const auto keys = static_cast<std::size_t>(program.layers) * kTensorCount + kStageCount;
    std::vector<std::vector<std::array<int, 2>>> writers(keys);
    for (const auto& task : tasks)
        for (const auto region : task.owned)
            writers[region.key].push_back({region.begin, region.end});
    auto covered = [&](int key, int width) {
        auto& entries = writers[key];
        std::sort(entries.begin(), entries.end());
        int end = 0;
        for (const auto [begin, finish] : entries) {
            require(begin == end, "missing or overlapping output rows");
            end = finish;
        }
        require(end == width, "incomplete output ownership");
    };
    constexpr std::array produced{Id::u, Id::q, Id::k, Id::v, Id::q_rot, Id::K, Id::V,
        Id::attention, Id::r, Id::z, Id::gate, Id::up, Id::product, Id::y};
    for (int layer = 0; layer < program.layers; ++layer)
        for (Id tensor : produced)
            covered(layer * static_cast<int>(kTensorCount) + static_cast<int>(tensor_index(tensor)),
                program.width(tensor));
    for (Stage stage : {Stage::embedding, Stage::final_norm, Stage::logits}) {
        if (program.full_stack)
            covered(program.stage_key(stage), program.stage_width(stage));
        else
            require(writers[program.stage_key(stage)].empty(), "stack stage in a block package");
    }
}

// Executes the schedule symbolically, one wave at a time. A read must observe
// exactly the value its logical producer wrote in an earlier wave; within one
// wave, two tasks may not write the same element or write an element another
// task reads. Grid barriers separate waves, so this is sufficient for the
// worker to compute every value from the intended inputs.
class Dataflow
{
  public:
    explicit Dataflow(const Program& program)
    {
        const auto& s = program.s;
        cells_.resize(Buffers::kv + 2 * static_cast<std::size_t>(program.layers));
        // A block package receives h_0 as input; a stack writes it by embedding.
        cells_[Buffers::hidden].assign(s.d, {program.full_stack ? -1 : 0});
        cells_[Buffers::hidden + 1].assign(s.d, {-1});
        for (std::size_t id = 0; id < kTensorCount; ++id)
            cells_[Buffers::scratch + id].assign(program.width(static_cast<Id>(id)), {});
        cells_[Buffers::scores].assign(s.nq, {});
        cells_[Buffers::logits].assign(program.vocab, {});
        for (int layer = 0; layer < program.layers; ++layer)
            for (int value = 0; value < 2; ++value)
                cells_[Buffers::kv + 2 * layer + value].assign(s.nkv * s.dh, {});
    }

    void run_wave(int wave, const std::vector<int>& ids, const std::vector<CheckedTask>& tasks)
    {
        constexpr int many = -2;
        for (const int id : ids)
            for (const auto access : tasks[id].reads)
                for (auto& cell : range(access)) {
                    require(cell.version == access.version, "read does not see its producer");
                    if (cell.read_wave != wave) {
                        cell.read_wave = wave;
                        cell.read_task = id;
                    } else if (cell.read_task != id) {
                        cell.read_task = many;
                    }
                }
        for (const int id : ids)
            for (const auto access : tasks[id].writes)
                for (auto& cell : range(access)) {
                    require(cell.write_wave != wave || cell.write_task == id,
                        "tasks in one wave write the same element");
                    require(cell.read_wave != wave || cell.read_task == id,
                        "a task overwrites an element another task reads in the same wave");
                    cell.write_wave = wave;
                    cell.write_task = id;
                    cell.version = access.version;
                }
    }

  private:
    struct Cell
    {
        int version = 0;
        int write_wave = -1;
        int write_task = -1;
        int read_wave = -1;
        int read_task = -1;
    };

    std::span<Cell> range(const Access& access)
    {
        auto& cells = cells_[access.buffer];
        return std::span(cells).subspan(access.begin, access.end - access.begin);
    }

    std::vector<std::vector<Cell>> cells_;
};
} // namespace

void validate_program(const json& document, const PlanShape& s, int workers, LoadedPlan& result)
{
    const Program program{s, result.stack.layers, result.stack.vocab_size, result.stack.full_stack,
        result.weight_dtype == WeightDtype::bfloat16};
    const auto& input = document.at("tasks");
    const auto& waves = document.at("waves");
    require(input.is_array() && !input.empty() &&
                input.size() < static_cast<std::size_t>(std::numeric_limits<int>::max()),
        "invalid task count");
    require(
        waves.is_array() && !waves.empty() && waves.size() <= input.size(), "invalid wave count");
    const int count = static_cast<int>(input.size());
    result.task_count = count;
    result.wave_count = static_cast<int>(waves.size());
    std::vector<CheckedTask> tasks(count);
    std::vector<bool> seen(count);
    for (const auto& task : input) {
        const int id = index(task.at("id"), count);
        require(!seen[id], "duplicate task ID");
        seen[id] = true;
        tasks[id] = program.check(task, workers, result.wave_count);
    }
    check_ownership(program, tasks);

    std::fill(seen.begin(), seen.end(), false);
    std::vector<std::vector<TaskRecord>> streams(workers);
    std::vector<std::vector<std::int32_t>> stream_tasks(workers);
    TaskRecord barrier{};
    barrier.opcode = Opcode::barrier;
    barrier.layer = kNoLayer;
    Dataflow dataflow(program);
    for (int wave = 0; wave < result.wave_count; ++wave) {
        const auto& entry = waves[wave];
        require(index(entry.at("id"), result.wave_count) == wave &&
                    entry.at("workers").is_array() &&
                    entry.at("workers").size() == static_cast<std::size_t>(workers),
            "invalid wave workers");
        std::vector<std::int64_t> loads(workers);
        std::vector<int> members;
        for (int worker = 0; worker < workers; ++worker) {
            const auto& lane = entry.at("workers")[worker];
            require(lane.is_array(), "invalid worker task list");
            for (const auto& item : lane) {
                const int id = index(item, count);
                require(!seen[id] && tasks[id].wave == wave && tasks[id].worker == worker,
                    "task is duplicated or assigned to a different wave/worker");
                seen[id] = true;
                require(tasks[id].work <= std::numeric_limits<std::int64_t>::max() - loads[worker],
                    "worker work estimate overflow");
                loads[worker] += tasks[id].work;
                streams[worker].push_back(tasks[id].record);
                stream_tasks[worker].push_back(id);
                members.push_back(id);
            }
            // Every worker reaches the same barriers, including idle workers.
            if (wave + 1 < result.wave_count) {
                streams[worker].push_back(barrier);
                stream_tasks[worker].push_back(-1);
            }
        }
        dataflow.run_wave(wave, members, tasks);
        result.worker_loads.push_back(std::move(loads));
    }
    require(std::all_of(seen.begin(), seen.end(), [](bool value) { return value; }),
        "task is absent from the schedule");
    for (int worker = 0; worker < workers; ++worker) {
        const auto& stream = streams[worker];
        require(
            result.instructions.size() + stream.size() <= std::numeric_limits<std::uint32_t>::max(),
            "too many worker instructions");
        result.streams.push_back({static_cast<std::uint32_t>(result.instructions.size()),
            static_cast<std::uint32_t>(stream.size())});
        result.instructions.insert(result.instructions.end(), stream.begin(), stream.end());
        result.instruction_tasks.insert(result.instruction_tasks.end(),
            stream_tasks[worker].begin(), stream_tasks[worker].end());
    }
}
} // namespace tileinfer::detail
