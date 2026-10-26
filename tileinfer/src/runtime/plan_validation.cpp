#include "detail/plan_validation.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
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
    selection, // one greedy candidate slot per logits task
};
constexpr int kStageCount = 4;

// Attention workspaces, indexed per (chunk, query head): element chunk * nq + head.
enum class Workspace
{
    scores,   // the chunk's scores, later probabilities
    stats,    // the chunk's softmax maximum and sum
    partials, // the chunk's partial P*V
};
constexpr int kWorkspaceCount = 3;

// Physical storage the worker reads or writes while executing one token.
// Scratch and the attention and FFN workspaces are shared by all layers, the
// two hidden buffers alternate, and each layer appends its own K/V row.
struct Buffers
{
    static constexpr int hidden = 0;  // hidden + (h % 2)
    static constexpr int scratch = 2; // scratch + tensor index
    static constexpr int workspace = scratch + static_cast<int>(kTensorCount);
    static constexpr int ffn = workspace + kWorkspaceCount; // FFN down partials, chunk * d + row
    static constexpr int logits = ffn + 1;
    static constexpr int selection = logits + 1; // greedy candidates, one per slot
    static constexpr int kv = selection + 1;     // kv + 2 * layer + (0 for K, 1 for V)
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
    int key; // per layer: tensor index or workspace; stage keys follow all layers
    int begin;
    int end;
};

// Attention runs in one of two regimes per position. Split: several chunks
// are active, scores and values tasks pass scores and statistics through the
// workspaces. Fused: one chunk is active, the chunk-0 scores task computes
// the partials itself, values tasks do nothing, and the conditional barrier
// between the two waves is skipped. Positions with fewer active chunks than
// the program holds only drop accesses of the all-chunks split regime, so
// checking these two covers every position.
enum class Regime
{
    split,
    fused,
};

struct CheckedTask
{
    TaskRecord record{};
    int wave;
    int worker;
    std::int64_t work;
    std::vector<Access> reads;
    std::vector<Access> writes;
    std::vector<Region> owned; // split regime, as the plan declares
    bool fused_differs = false;
    std::vector<Access> fused_reads;
    std::vector<Access> fused_writes;

    [[nodiscard]] const std::vector<Access>& reads_in(Regime regime) const
    {
        return regime == Regime::fused && fused_differs ? fused_reads : reads;
    }
    [[nodiscard]] const std::vector<Access>& writes_in(Regime regime) const
    {
        return regime == Regime::fused && fused_differs ? fused_writes : writes;
    }
};

// Per layer: every tensor, every attention workspace, then the FFN partials.
constexpr int kFfnKey = static_cast<int>(kTensorCount) + kWorkspaceCount;
constexpr int kLayerKeys = kFfnKey + 1;

struct Program
{
    const ModelShape& s;
    int layers;
    int vocab;
    int slots; // greedy candidate slots
    bool full_stack;
    bool bf16;

    [[nodiscard]] int width(Id tensor) const
    {
        if (tensor == Id::K || tensor == Id::V || tensor == Id::k)
            return s.kv_heads * s.head_dim;
        return tensor == Id::product ? s.intermediate_size : s.hidden_size;
    }

    [[nodiscard]] int stage_key(Stage stage) const
    {
        return layers * kLayerKeys + static_cast<int>(stage);
    }

    [[nodiscard]] int stage_width(Stage stage) const
    {
        if (stage == Stage::selection)
            return slots;
        return stage == Stage::logits ? vocab : s.hidden_size;
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
    const int input_hidden = Buffers::hidden + layer % 2;
    const int output_hidden = Buffers::hidden + (layer + 1) % 2;
    const int produced = layer + 1;
    const int d = s.hidden_size;
    const int dh = s.head_dim;
    const int kv_width = s.kv_heads * dh;
    const int group = s.query_heads / s.kv_heads;
    const int lanes = s.attention_chunks * s.query_heads; // workspace elements: chunk * nq + head

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
            {layer * kLayerKeys + static_cast<int>(tensor_index(id)), begin, end});
        if (listed)
            outputs.push_back({{"tensor", tensor_index(id)}, {"rows", {begin, end}}});
    };
    auto workspace = [&](Workspace kind) { return Buffers::workspace + static_cast<int>(kind); };
    auto rows = [&](int limit) {
        require(op.at("rows").is_array() && op.at("rows").size() == 2, "invalid rows");
        const int begin = index(op.at("rows")[0], limit);
        const int end = index(op.at("rows")[1], static_cast<std::int64_t>(limit) + 1);
        require(end > begin, "empty or reversed task rows");
        result.record.begin = begin;
        result.record.end = end;
        expected = {{"rows", {begin, end}}};
        return std::pair{begin, end};
    };
    auto stage_output = [&](Stage kind, const char* name, int begin, int end) {
        result.owned.push_back({stage_key(kind), begin, end});
        outputs.push_back({{"stage", name}, {"rows", {begin, end}}});
    };
    // Attention tasks name one K/V head and context chunk; return its query heads.
    auto attention_chunk = [&](Workspace kind, const char* name) {
        const int kv_head = index(op.at("kv_head"), s.kv_heads);
        const int chunk = index(op.at("chunk"), s.attention_chunks);
        expected = {{"kv_head", kv_head}, {"chunk", chunk}};
        result.record.kv_head = kv_head;
        result.record.chunk = chunk;
        const int first = chunk * s.query_heads + kv_head * group;
        result.owned.push_back(
            {layer * kLayerKeys + static_cast<int>(kTensorCount) + static_cast<int>(kind), first,
                first + group});
        outputs.push_back({{"workspace", name}, {"chunk", chunk},
            {"heads", {kv_head * group, (kv_head + 1) * group}}});
        return std::tuple{kv_head, chunk, first};
    };

    if (code == "embedding") {
        result.record.opcode = Opcode::embedding;
        const auto [begin, end] = rows(d);
        result.writes.push_back({Buffers::hidden, begin, end, 0});
        stage_output(Stage::embedding, "embedding", begin, end);
        result.work = end - begin;
    } else if (code == "final_rms_norm") {
        result.record.opcode = Opcode::final_rms_norm;
        const auto [begin, end] = rows(d);
        result.reads.push_back({Buffers::hidden + layers % 2, 0, d, layers});
        result.writes.push_back({Buffers::hidden + (layers + 1) % 2, begin, end, layers + 1});
        stage_output(Stage::final_norm, "final_norm", begin, end);
        result.work = d + end - begin;
    } else if (code == "logits") {
        // Vocabulary rows, then their greedy candidate. The worker combines
        // every slot before the next token's embedding, after a grid barrier.
        result.record.opcode = Opcode::logits;
        const auto [begin, end] = rows(vocab);
        const int slot = index(op.at("chunk"), slots);
        expected["chunk"] = slot;
        result.record.chunk = slot;
        result.reads.push_back({Buffers::hidden + (layers + 1) % 2, 0, d, layers + 1});
        result.writes.push_back({Buffers::logits, begin, end, 1});
        result.writes.push_back({Buffers::selection, slot, slot + 1, 1});
        stage_output(Stage::logits, "logits", begin, end);
        result.owned.push_back({stage_key(Stage::selection), slot, slot + 1});
        outputs.push_back({{"workspace", "selection"}, {"chunk", slot}});
        result.work = static_cast<std::int64_t>(end - begin) * d;
    } else if (code == "qkv") {
        // Rows of the concatenated [Q; K; V] projection: Q and K unrotated
        // into scratch, V into the cache row of the current position.
        result.record.opcode = Opcode::qkv;
        const auto [begin, end] = rows(d + 2 * kv_width);
        read(Id::x, 0, d);
        if (begin < d)
            write(Id::q, begin, std::min(end, d));
        if (begin < d + kv_width && end > d)
            write(Id::k, std::max(begin, d) - d, std::min(end, d + kv_width) - d);
        if (end > d + kv_width) {
            const int first = std::max(begin, d + kv_width) - d - kv_width;
            const int last = end - d - kv_width;
            write(Id::V, first, last, false);
            outputs.push_back({{"tensor", tensor_index(Id::V)}, {"position", "current"},
                {"rows", {first, last}}});
        }
        result.work = static_cast<std::int64_t>(end - begin) * d;
    } else if (code == "attention_scores") {
        result.record.opcode = Opcode::attention_scores;
        const auto [kv_head, chunk, first] = attention_chunk(Workspace::scores, "scores");
        read(Id::q, kv_head * group * dh, (kv_head + 1) * group * dh);
        read(Id::k, kv_head * dh, (kv_head + 1) * dh);
        for (Workspace kind : {Workspace::scores, Workspace::stats})
            result.writes.push_back({workspace(kind), first, first + group, produced});
        // Every task rotates its heads; chunk 0 stores rotated Q and the rotated
        // K row. The task whose chunk ends at the current position takes that
        // row from scratch, so no task reads the cache row during the token.
        if (chunk == 0) {
            write(Id::q_rot, kv_head * group * dh, (kv_head + 1) * group * dh);
            write(Id::K, kv_head * dh, (kv_head + 1) * dh, false);
            outputs.push_back(
                {{"tensor", tensor_index(Id::K)}, {"position", "current"}, {"head", kv_head}});
        }
        result.fused_differs = true;
        if (chunk == 0) {
            result.fused_reads = result.reads;
            result.fused_reads.push_back(
                {buffer(Id::V), kv_head * dh, (kv_head + 1) * dh, version(Id::V)});
            result.fused_writes.assign(result.writes.end() - 2, result.writes.end());
            result.fused_writes.push_back(
                {workspace(Workspace::partials), first, first + group, produced});
        }
        result.work = 2LL * std::min(s.attention_chunk_steps, s.max_context) * dh * group;
    } else if (code == "attention_values") {
        result.record.opcode = Opcode::attention_values;
        const auto [kv_head, chunk, first] = attention_chunk(Workspace::partials, "partials");
        // The softmax combines every chunk's statistics for these heads.
        for (int c = 0; c < s.attention_chunks; ++c) {
            const int heads = c * s.query_heads + kv_head * group;
            result.reads.push_back({workspace(Workspace::stats), heads, heads + group, produced});
        }
        result.reads.push_back({workspace(Workspace::scores), first, first + group, produced});
        read(Id::V, kv_head * dh, (kv_head + 1) * dh);
        result.writes.push_back({workspace(Workspace::scores), first, first + group, produced});
        result.writes.push_back({workspace(Workspace::partials), first, first + group, produced});
        result.fused_differs = true; // nothing to do
        result.work = 2LL * std::min(s.attention_chunk_steps, s.max_context) * dh * group;
    } else if (code == "output_projection") {
        result.record.opcode = Opcode::output_projection;
        const auto [begin, end] = rows(d);
        result.reads.push_back({workspace(Workspace::partials), 0, lanes, produced});
        read(Id::x, begin, end);
        write(Id::r, begin, end);
        // The task owning row 0 also stores the combined attention vector.
        if (begin == 0)
            write(Id::attention, 0, d);
        result.fused_differs = true;
        result.fused_reads = result.reads;
        result.fused_reads.front().end = s.query_heads; // chunk 0 only
        result.fused_writes = result.writes;
        result.work = static_cast<std::int64_t>(end - begin) * d;
    } else if (code == "ffn") {
        // A product chunk's gate/up rows, then their FP32 down partial over
        // every output. The task reads back only its own product rows, which
        // ops::down_partial stages in the shared vector.
        result.record.opcode = Opcode::ffn;
        const auto [begin, end] = rows(s.intermediate_size);
        const int chunk = index(op.at("chunk"), s.ffn_chunks);
        expected["chunk"] = chunk;
        result.record.chunk = chunk;
        require(static_cast<std::int64_t>(end - begin) * (bf16 ? 2 : 4) <= kAttentionSharedBytes,
            "FFN chunk exceeds the compiled worker's shared memory");
        read(Id::r, 0, d);
        write(Id::product, begin, end);
        result.writes.push_back({Buffers::ffn, chunk * d, (chunk + 1) * d, produced});
        result.owned.push_back({layer * kLayerKeys + kFfnKey, chunk * d, (chunk + 1) * d});
        outputs.push_back({{"workspace", "ffn_partials"}, {"chunk", chunk}});
        result.work = 3LL * (end - begin) * d;
    } else if (code == "ffn_reduce") {
        // Rows of y: every chunk's partial for these rows, in chunk order.
        result.record.opcode = Opcode::ffn_reduce;
        const auto [begin, end] = rows(d);
        for (int chunk = 0; chunk < s.ffn_chunks; ++chunk)
            result.reads.push_back({Buffers::ffn, chunk * d + begin, chunk * d + end, produced});
        read(Id::r, begin, end);
        write(Id::y, begin, end);
        result.work = static_cast<std::int64_t>(end - begin) * s.ffn_chunks;
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
    const auto keys = static_cast<std::size_t>(program.layers) * kLayerKeys + kStageCount;
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
    constexpr std::array produced{
        Id::q, Id::k, Id::q_rot, Id::K, Id::V, Id::attention, Id::r, Id::product, Id::y};
    const int lanes = program.s.attention_chunks * program.s.query_heads;
    for (int layer = 0; layer < program.layers; ++layer) {
        for (Id tensor : produced)
            covered(
                layer * kLayerKeys + static_cast<int>(tensor_index(tensor)), program.width(tensor));
        for (Workspace kind : {Workspace::scores, Workspace::partials})
            covered(layer * kLayerKeys + static_cast<int>(kTensorCount) + static_cast<int>(kind),
                lanes);
        covered(layer * kLayerKeys + kFfnKey, program.s.ffn_chunks * program.s.hidden_size);
    }
    for (Stage stage : {Stage::embedding, Stage::final_norm, Stage::logits, Stage::selection}) {
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
        cells_[Buffers::hidden].assign(s.hidden_size, {program.full_stack ? -1 : 0});
        cells_[Buffers::hidden + 1].assign(s.hidden_size, {-1});
        for (std::size_t id = 0; id < kTensorCount; ++id)
            cells_[Buffers::scratch + id].assign(program.width(static_cast<Id>(id)), {});
        for (int kind = 0; kind < kWorkspaceCount; ++kind)
            cells_[Buffers::workspace + kind].assign(s.attention_chunks * s.query_heads, {});
        cells_[Buffers::ffn].assign(static_cast<std::size_t>(s.ffn_chunks) * s.hidden_size, {});
        cells_[Buffers::logits].assign(program.vocab, {});
        cells_[Buffers::selection].assign(program.slots, {});
        for (int layer = 0; layer < program.layers; ++layer)
            for (int value = 0; value < 2; ++value)
                cells_[Buffers::kv + 2 * layer + value].assign(s.kv_heads * s.head_dim, {});
    }

    void run_wave(
        int wave, const std::vector<int>& ids, const std::vector<CheckedTask>& tasks, Regime regime)
    {
        constexpr int many = -2;
        for (const int id : ids)
            for (const auto access : tasks[id].reads_in(regime))
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
            for (const auto access : tasks[id].writes_in(regime))
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

void validate_program(const json& document, const ModelShape& s, int workers, LoadedPlan& result)
{
    const Program program{s, result.stack.layers, result.stack.vocab_size, result.selection_slots,
        result.stack.full_stack, result.weight_dtype == WeightDtype::bfloat16};
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
        // A multi-token launch combines the candidates before the first wave,
        // so a barrier must separate their writes from that read.
        require(tasks[id].record.opcode != Opcode::logits || tasks[id].wave > 0,
            "logits task in the first wave");
    }
    check_ownership(program, tasks);

    // A wave of only scores tasks followed by a wave of only values tasks ends
    // with a conditional barrier, and every other wave with a plain one.
    std::vector<int> wave_opcode(result.wave_count, -1); // the opcode all members share, or -2
    for (const auto& task : tasks) {
        auto& code = wave_opcode[task.wave];
        const int opcode = static_cast<int>(task.record.opcode);
        code = code == -1 || code == opcode ? opcode : -2;
    }
    auto conditional = [&](int wave) {
        return wave + 1 < result.wave_count &&
               wave_opcode[wave] == static_cast<int>(Opcode::attention_scores) &&
               wave_opcode[wave + 1] == static_cast<int>(Opcode::attention_values);
    };

    std::fill(seen.begin(), seen.end(), false);
    std::vector<std::vector<TaskRecord>> streams(workers);
    std::vector<std::vector<std::int32_t>> stream_tasks(workers);
    Dataflow split(program);
    Dataflow fused(program);
    std::vector<int> fused_members;
    for (int wave = 0; wave < result.wave_count; ++wave) {
        const auto& entry = waves[wave];
        require(index(entry.at("id"), result.wave_count) == wave &&
                    entry.at("workers").is_array() &&
                    entry.at("workers").size() == static_cast<std::size_t>(workers),
            "invalid wave workers");
        const json expected_condition = {{"skip_up_to_length", s.attention_fused_steps}};
        require(conditional(wave) ? entry.contains("conditional_barrier") &&
                                        entry.at("conditional_barrier") == expected_condition
                                  : !entry.contains("conditional_barrier"),
            "invalid conditional barrier");
        TaskRecord barrier{};
        barrier.opcode = Opcode::barrier;
        barrier.layer = kNoLayer;
        barrier.begin = conditional(wave) ? s.attention_fused_steps : 0;
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
        split.run_wave(wave, members, tasks, Regime::split);
        // Without the conditional barrier, its two waves run as one.
        fused_members.insert(fused_members.end(), members.begin(), members.end());
        if (!conditional(wave)) {
            fused.run_wave(wave, fused_members, tasks, Regime::fused);
            fused_members.clear();
        }
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
