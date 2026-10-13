#include "plan.hpp"

#include <picosha2.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <utility>

namespace tileinfer {
namespace {
using json = nlohmann::json;

void require(bool condition, std::string_view message)
{
    if (!condition)
        throw std::runtime_error(std::string(message));
}

std::vector<std::uint8_t> read_bytes(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    require(static_cast<bool>(file), "cannot open " + path.string());
    const auto size = file.tellg();
    require(size >= 0, "cannot size " + path.string());
    std::vector<std::uint8_t> result(static_cast<size_t>(size));
    file.seekg(0);
    if (!result.empty())
        file.read(reinterpret_cast<char*>(result.data()), size);
    require(static_cast<bool>(file), "cannot read " + path.string());
    return result;
}

int positive(const json& value, const char* name)
{
    require(value.is_number_integer(), std::string("invalid ") + name);
    const auto n = value.get<std::int64_t>();
    require(n > 0 && n <= std::numeric_limits<int>::max(), std::string("invalid ") + name);
    return static_cast<int>(n);
}

struct Spec
{
    int d;
    int f;
    int nq;
    int nkv;
    int dh;
    int context;
    int tile;
};

constexpr std::array<const char*, 24> names{"x", "attn_norm", "u", "Wq", "q", "Wk", "k", "Wv", "v",
    "q_rot", "K", "V", "attention", "Wo", "r", "ffn_norm", "z", "Wgate", "gate", "Wup", "up",
    "product", "Wdown", "y"};
constexpr std::array<int, 9> weight_ids{1, 3, 5, 7, 13, 15, 17, 19, 22};
constexpr std::array<int, 11> scratch_ids{2, 4, 6, 8, 9, 12, 14, 16, 18, 20, 21};
constexpr std::array<int, 2> kv_ids{10, 11};

std::vector<int> shape(const Spec& s, int id)
{
    const int kv_width = s.nkv * s.dh;
    switch (id) {
    case 1:
    case 15:
        return {s.d};
    case 3:
    case 13:
        return {s.d, s.d};
    case 5:
    case 7:
        return {kv_width, s.d};
    case 17:
    case 19:
        return {s.f, s.d};
    case 22:
        return {s.d, s.f};
    case 6:
    case 8:
        return {kv_width};
    case 18:
    case 20:
    case 21:
        return {s.f};
    case 10:
    case 11:
        return {1, s.context, s.nkv, s.dh};
    default:
        return {s.d};
    }
}

std::string storage_class(int id)
{
    if (std::find(weight_ids.begin(), weight_ids.end(), id) != weight_ids.end())
        return "weight";
    if (id == 10 || id == 11)
        return "kv";
    if (id == 0)
        return "input";
    if (id == 23)
        return "output";
    return "scratch";
}

json descriptors(const Spec& s, std::string_view weight_dtype)
{
    json result = json::array();
    for (int id = 0; id < static_cast<int>(names.size()); ++id) {
        const auto dims = shape(s, id);
        std::vector<int> strides(dims.size(), 1);
        for (int index = static_cast<int>(dims.size()) - 2; index >= 0; --index)
            strides[index] = strides[index + 1] * dims[index + 1];
        result.push_back({{"id", id}, {"name", names[id]}, {"shape", dims}, {"strides", strides},
            {"dtype", storage_class(id) == "weight" ? weight_dtype : "float32"},
            {"storage", storage_class(id)}});
    }
    return result;
}

template <std::size_t N>
std::pair<json, std::uint64_t> regions(
    const Spec& s, const std::array<int, N>& ids, int alignment, int element_bytes = 4)
{
    json result = json::array();
    std::uint64_t cursor = 0;
    for (int id : ids) {
        cursor += (alignment - cursor % alignment) % alignment;
        std::uint64_t bytes = element_bytes;
        for (int dimension : shape(s, id))
            bytes *= static_cast<std::uint64_t>(dimension);
        result.push_back(
            {{"tensor", id}, {"offset", cursor}, {"bytes", bytes}, {"alignment", alignment}});
        cursor += bytes;
    }
    return {result, cursor};
}

using Region = std::tuple<int, int, int>;
using WriterMap = std::map<std::string, std::vector<Region>>;

std::vector<json> graph(const Spec& s)
{
    std::vector<json> tasks;
    WriterMap writers;
    auto add = [&](const char* opcode, json operands, json output, std::vector<int> predecessors,
                   std::int64_t work) {
        std::sort(predecessors.begin(), predecessors.end());
        predecessors.erase(
            std::unique(predecessors.begin(), predecessors.end()), predecessors.end());
        const int id = static_cast<int>(tasks.size());
        tasks.push_back({{"id", id}, {"opcode", opcode}, {"operands", std::move(operands)},
            {"output", std::move(output)}, {"predecessors", predecessors},
            {"estimated_work", work}});
        return id;
    };
    auto owners = [&](const std::string& name, int begin, int end) {
        std::vector<int> result;
        for (auto [lo, hi, id] : writers.at(name))
            if (lo < end && begin < hi)
                result.push_back(id);
        return result;
    };
    auto tiled = [&](const char* input, int input_id, int weight_id, const char* output,
                     int output_id, int rows, int width, const std::vector<int>& predecessors,
                     int addend) {
        writers[output] = {};
        for (int begin = 0; begin < rows; begin += s.tile) {
            const int end = std::min(begin + s.tile, rows);
            const int id = add("gemv_f32",
                {{"input", input_id}, {"weight", weight_id}, {"output", output_id},
                    {"rows", {begin, end}}, {"addend", addend < 0 ? json(nullptr) : json(addend)}},
                json::array({{{"tensor", output_id}, {"rows", {begin, end}}}}), predecessors,
                static_cast<std::int64_t>(end - begin) * width);
            writers[output].push_back({begin, end, id});
        }
        (void)input;
    };
    auto ids = [&](const std::string& name) {
        std::vector<int> result;
        for (auto [begin, end, id] : writers.at(name)) {
            (void)begin;
            (void)end;
            result.push_back(id);
        }
        return result;
    };
    auto combine = [](std::vector<int> first, const std::vector<int>& second) {
        first.insert(first.end(), second.begin(), second.end());
        return first;
    };

    const int kv_width = s.nkv * s.dh;
    const int first = add("rms_norm_f32", {{"input", 0}, {"scale", 1}, {"output", 2}},
        json::array({{{"tensor", 2}, {"rows", {0, s.d}}}}), {}, s.d);
    writers["u"] = {{0, s.d, first}};
    tiled("u", 2, 3, "q", 4, s.d, s.d, {first}, -1);
    tiled("u", 2, 5, "k", 6, kv_width, s.d, {first}, -1);
    tiled("u", 2, 7, "v", 8, kv_width, s.d, {first}, -1);
    writers["q_rot"] = {};
    for (int head = 0; head < s.nq; ++head) {
        const int begin = head * s.dh;
        const int end = begin + s.dh;
        const int id = add("rope_query_f32", {{"input", 4}, {"output", 9}, {"head", head}},
            json::array({{{"tensor", 9}, {"rows", {begin, end}}}}), owners("q", begin, end), s.dh);
        writers["q_rot"].push_back({begin, end, id});
    }
    std::vector<int> cache_tasks;
    for (int head = 0; head < s.nkv; ++head) {
        const int begin = head * s.dh;
        const int end = begin + s.dh;
        const int id = add("rope_cache_f32",
            {{"key", 6}, {"value", 8}, {"key_cache", 10}, {"value_cache", 11}, {"kv_head", head}},
            json::array({{{"tensor", 10}, {"position", "current"}, {"head", head}},
                {{"tensor", 11}, {"position", "current"}, {"head", head}}}),
            combine(owners("k", begin, end), owners("v", begin, end)), 2 * s.dh);
        cache_tasks.push_back(id);
    }
    writers["attention"] = {};
    for (int head = 0; head < s.nq; ++head) {
        const int kv_head = head / (s.nq / s.nkv);
        const int begin = head * s.dh;
        const int end = begin + s.dh;
        auto predecessors = owners("q_rot", begin, end);
        predecessors.push_back(cache_tasks[kv_head]);
        const int id = add("attention_prefix_f32",
            {{"query", 9}, {"key_cache", 10}, {"value_cache", 11}, {"output", 12}, {"head", head},
                {"kv_head", kv_head}},
            json::array({{{"tensor", 12}, {"rows", {begin, end}}}}), predecessors,
            static_cast<std::int64_t>(2) * s.context * s.dh);
        writers["attention"].push_back({begin, end, id});
    }
    tiled("attention", 12, 13, "r", 14, s.d, s.d, ids("attention"), 0);
    const int second = add("rms_norm_f32", {{"input", 14}, {"scale", 15}, {"output", 16}},
        json::array({{{"tensor", 16}, {"rows", {0, s.d}}}}), ids("r"), s.d);
    writers["z"] = {{0, s.d, second}};
    tiled("z", 16, 17, "gate", 18, s.f, s.d, {second}, -1);
    tiled("z", 16, 19, "up", 20, s.f, s.d, {second}, -1);
    writers["product"] = {};
    for (int begin = 0; begin < s.f; begin += s.tile) {
        const int end = std::min(begin + s.tile, s.f);
        const int id = add("silu_product_f32",
            {{"gate", 18}, {"up", 20}, {"output", 21}, {"rows", {begin, end}}},
            json::array({{{"tensor", 21}, {"rows", {begin, end}}}}),
            combine(owners("gate", begin, end), owners("up", begin, end)), end - begin);
        writers["product"].push_back({begin, end, id});
    }
    tiled("product", 21, 22, "y", 23, s.d, s.f, ids("product"), 14);
    return tasks;
}

json schedule(std::vector<json>& tasks, int workers)
{
    std::vector<std::vector<int>> buckets;
    for (auto& task : tasks) {
        int wave = 0;
        for (const auto& predecessor : task.at("predecessors"))
            wave = std::max(wave, 1 + tasks.at(predecessor.get<int>()).at("wave").get<int>());
        task["wave"] = wave;
        if (buckets.size() <= static_cast<size_t>(wave))
            buckets.resize(wave + 1);
        buckets[wave].push_back(task.at("id").get<int>());
    }
    json result = json::array();
    for (int wave = 0; wave < static_cast<int>(buckets.size()); ++wave) {
        std::vector<std::vector<int>> lanes(workers);
        std::vector<std::int64_t> loads(workers);
        auto ids = buckets[wave];
        std::sort(ids.begin(), ids.end(), [&](int a, int b) {
            const auto wa = tasks[a].at("estimated_work").get<std::int64_t>();
            const auto wb = tasks[b].at("estimated_work").get<std::int64_t>();
            return wa == wb ? a < b : wa > wb;
        });
        for (int id : ids) {
            const int worker =
                static_cast<int>(std::min_element(loads.begin(), loads.end()) - loads.begin());
            tasks[id]["worker"] = worker;
            lanes[worker].push_back(id);
            loads[worker] += tasks[id].at("estimated_work").get<std::int64_t>();
        }
        result.push_back({{"id", wave}, {"workers", lanes}, {"estimated_work", loads}});
    }
    return result;
}

TaskRecord encode(const json& task)
{
    TaskRecord result{};
    const auto& op = task.at("operands");
    const auto code = task.at("opcode").get<std::string>();
    if (code == "rms_norm_f32") {
        result.opcode = Opcode::rms_norm_f32;
        result.a = op.at("input");
        result.b = op.at("scale");
        result.c = op.at("output");
    } else if (code == "gemv_f32") {
        result.opcode = Opcode::gemv_f32;
        result.a = op.at("input");
        result.b = op.at("weight");
        result.c = op.at("output");
        result.d = op.at("addend").is_null() ? std::numeric_limits<std::uint32_t>::max()
                                             : op.at("addend").get<unsigned>();
        result.begin = op.at("rows")[0];
        result.end = op.at("rows")[1];
    } else if (code == "rope_query_f32") {
        result.opcode = Opcode::rope_query_f32;
        result.a = op.at("input");
        result.b = op.at("output");
        result.head = op.at("head");
    } else if (code == "rope_cache_f32") {
        result.opcode = Opcode::rope_cache_f32;
        result.a = op.at("key");
        result.b = op.at("value");
        result.c = op.at("key_cache");
        result.d = op.at("value_cache");
        result.kv_head = op.at("kv_head");
    } else if (code == "attention_prefix_f32") {
        result.opcode = Opcode::attention_prefix_f32;
        result.a = op.at("query");
        result.b = op.at("key_cache");
        result.c = op.at("value_cache");
        result.d = op.at("output");
        result.head = op.at("head");
        result.kv_head = op.at("kv_head");
    } else if (code == "silu_product_f32") {
        result.opcode = Opcode::silu_product_f32;
        result.a = op.at("gate");
        result.b = op.at("up");
        result.c = op.at("output");
        result.begin = op.at("rows")[0];
        result.end = op.at("rows")[1];
    } else {
        throw std::runtime_error("unsupported decoder opcode");
    }
    return result;
}

} // namespace

std::string sha256(const std::vector<std::uint8_t>& bytes)
{
    return picosha2::hash256_hex_string(bytes);
}

LoadedPlan load_plan(const std::string& package, const Target& device)
{
    const auto plan_data = read_bytes(std::filesystem::path(package) / "plan.json");
    const auto weights = read_bytes(std::filesystem::path(package) / "weights.bin");
    const auto document = json::parse(plan_data.begin(), plan_data.end());
    const auto& header = document.at("header");
    const auto model_kind = header.at("model_kind").get<std::string>();
    const bool full_stack = model_kind == "llama_decoder_stack_f32";
    const bool legacy = header.at("format_version") == 2;
    require(((legacy && header.at("executor_abi") == kLegacyAbi) ||
                (header.at("format_version") == 3 && header.at("executor_abi") == kAbi)) &&
                (model_kind == "llama_decoder_block_f32" || full_stack),
        "unsupported decoder plan header");
    const auto& model = document.at("model");
    const std::string weight_dtype =
        legacy ? "float32" : model.at("weight_dtype").get<std::string>();
    require(weight_dtype == "float32" || weight_dtype == "bfloat16", "unsupported weight dtype");
    const int weight_element_bytes = weight_dtype == "bfloat16" ? 2 : 4;
    Spec s{positive(model.at("hidden_size"), "hidden_size"),
        positive(model.at("intermediate_size"), "intermediate_size"),
        positive(model.at("num_attention_heads"), "num_attention_heads"),
        positive(model.at("num_key_value_heads"), "num_key_value_heads"),
        positive(model.at("head_dim"), "head_dim"),
        positive(model.at("max_context"), "max_context"),
        positive(model.at("tile_rows"), "tile_rows")};
    const int layers = positive(model.at("layers"), "layers");
    const int vocab = full_stack ? positive(model.at("vocab_size"), "vocab_size") : 0;
    const bool tied = full_stack ? model.at("tie_word_embeddings").get<bool>() : false;
    require(s.d == static_cast<std::int64_t>(s.nq) * s.dh && s.nq % s.nkv == 0 && s.dh % 2 == 0 &&
                (full_stack || layers == 1),
        "unsupported decoder dimensions");
    require(
        model.at("dtype") == "float32" && model.at("rope_type") == "llama3" &&
            model.at("rope_pairing") == "split_half" &&
            model.at("kv_cache_layout") == "[layers,max_context,kv_heads,head_dim]" &&
            model.at("rms_norm_eps").get<double>() > 0 &&
            model.at("rope_theta").get<double>() > 0 && model.at("rope_factor").get<double>() > 0 &&
            model.at("rope_low_freq_factor").get<double>() > 0 &&
            model.at("rope_high_freq_factor").get<double>() >
                model.at("rope_low_freq_factor").get<double>() &&
            positive(model.at("rope_original_max_position_embeddings"), "rope original context") >
                0,
        "unsupported decoder numeric semantics");
    json head_map = json::array();
    for (int head = 0; head < s.nq; ++head)
        head_map.push_back(head / (s.nq / s.nkv));
    require(
        model.at("head_to_kv") == head_map && model.at("tensors") == descriptors(s, weight_dtype),
        "invalid tensor descriptors or GQA mapping");

    const auto& target = document.at("target");
    const int workers = positive(target.at("worker_count"), "worker_count");
    require(target.at("compute_capability") == json::array({device.major, device.minor}) &&
                target.at("sm_count") == device.sm_count && workers == device.sm_count &&
                target.at("cooperative_launch") == true && device.cooperative_launch &&
                target.at("threads_per_block") == kThreads &&
                (legacy ? (target.at("kernel_fingerprint") == kLegacyFingerprint ||
                              target.at("kernel_fingerprint") == kLegacyProbeFingerprint)
                        : target.at("kernel_fingerprint") == kFingerprint) &&
                workers <= device.sm_count * device.active_blocks_per_sm,
        "decoder plan target does not match compiled worker");

    const auto& storage = document.at("storage");
    const auto [weight_regions, weight_bytes] = regions(s, weight_ids, 64, weight_element_bytes);
    const auto [scratch_regions, scratch_bytes] = regions(s, scratch_ids, 16);
    const auto [kv_regions, kv_bytes] = regions(s, kv_ids, 16);
    auto align64 = [](std::uint64_t value) { return value + (64 - value % 64) % 64; };
    StackConfig stack{layers, vocab, 0, 0, 0, 0, 0, 0, full_stack};
    const std::uint64_t expected_kv_bytes =
        full_stack ? kv_bytes * static_cast<std::uint64_t>(layers) : kv_bytes;
    require(storage.at("scratch") == scratch_regions &&
                storage.at("scratch_bytes") == scratch_bytes && storage.at("kv") == kv_regions &&
                storage.at("kv_bytes") == expected_kv_bytes &&
                storage.at("weights") == weight_regions,
        "invalid decoder tensor storage");
    if (full_stack) {
        const std::uint64_t embedding_bytes =
            static_cast<std::uint64_t>(vocab) * s.d * weight_element_bytes;
        const auto layer_offset = align64(embedding_bytes);
        const auto layer_stride = align64(weight_bytes);
        const auto norm_offset = align64(layer_offset + layers * layer_stride);
        const auto lm_offset =
            tied ? 0
                 : align64(norm_offset + static_cast<std::uint64_t>(s.d) * weight_element_bytes);
        const auto total =
            tied ? norm_offset + static_cast<std::uint64_t>(s.d) * weight_element_bytes
                 : lm_offset + embedding_bytes;
        require(total <= static_cast<std::uint64_t>(std::numeric_limits<int>::max()) *
                             weight_element_bytes &&
                    expected_kv_bytes <=
                        static_cast<std::uint64_t>(std::numeric_limits<int>::max()) * 4,
            "stack storage exceeds address range");
        const json expected_stack = {{"embedding_offset", 0}, {"embedding_bytes", embedding_bytes},
            {"layer_weights_offset", layer_offset}, {"layer_weight_stride", layer_stride},
            {"layer_weight_bytes", weight_bytes}, {"final_norm_offset", norm_offset},
            {"final_norm_bytes", static_cast<std::uint64_t>(s.d) * weight_element_bytes},
            {"lm_head_offset", lm_offset}, {"lm_head_alias", tied}, {"layer_kv_stride", kv_bytes}};
        require(storage.at("input_bytes") == 4 &&
                    storage.at("output_bytes") == static_cast<std::uint64_t>(vocab) * 4 &&
                    storage.at("weight_bytes") == total && storage.at("stack") == expected_stack &&
                    total == weights.size(),
            "invalid decoder stack storage");
        require(document.at("execution") ==
                    json{{"prelude", "embedding_lookup_f32"}, {"repeat_block_count", layers},
                        {"postlude", json::array({"rms_norm_f32", "logits_f32"})}},
            "invalid decoder stack execution stages");
        stack.embedding_offset = 0;
        stack.layer_weights_offset = static_cast<int>(layer_offset / weight_element_bytes);
        stack.layer_weight_stride = static_cast<int>(layer_stride / weight_element_bytes);
        stack.final_norm_offset = static_cast<int>(norm_offset / weight_element_bytes);
        stack.lm_head_offset = static_cast<int>(lm_offset / weight_element_bytes);
        stack.layer_kv_stride = static_cast<int>(kv_bytes / 4);
    } else {
        require(storage.at("input_bytes") == static_cast<std::uint64_t>(s.d) * 4 &&
                    storage.at("output_bytes") == static_cast<std::uint64_t>(s.d) * 4 &&
                    storage.at("weight_bytes") == weight_bytes && weight_bytes == weights.size(),
            "invalid decoder block storage");
    }
    const auto& integrity = document.at("integrity");
    require(integrity.at("weights_bytes") == weights.size() &&
                integrity.at("weights_sha256") == sha256(weights),
        "decoder weight checksum mismatch");

    auto expected_tasks = graph(s);
    const auto expected_waves = schedule(expected_tasks, workers);
    require(document.at("tasks") == expected_tasks && document.at("waves") == expected_waves,
        "invalid decoder task operands, ownership, dependencies, or schedule");

    LoadedPlan result{};
    result.weight_dtype = weight_dtype == "bfloat16" ? WeightDtype::bfloat16 : WeightDtype::float32;
    result.hidden_size = s.d;
    result.intermediate_size = s.f;
    result.query_heads = s.nq;
    result.kv_heads = s.nkv;
    result.head_dim = s.dh;
    result.max_context = s.context;
    result.workers = workers;
    result.stack = stack;
    result.numeric = {model.at("rms_norm_eps").get<float>(), model.at("rope_theta").get<float>(),
        model.at("rope_factor").get<float>(), model.at("rope_low_freq_factor").get<float>(),
        model.at("rope_high_freq_factor").get<float>(),
        model.at("rope_original_max_position_embeddings").get<float>()};
    result.wave_count = static_cast<int>(expected_waves.size());
    result.scratch_bytes = scratch_bytes;
    result.kv_bytes = expected_kv_bytes;
    result.weights = weights;
    result.plan_sha256 = sha256(plan_data);
    result.weights_sha256 = sha256(weights);
    result.tensors.resize(names.size());
    result.tensors[0] = {StorageClass::input, 0};
    result.tensors[23] = {StorageClass::output, 0};
    for (const auto& [entries, storage_id] : std::array<std::pair<json, StorageClass>, 3>{
             std::pair{weight_regions, StorageClass::weight},
             {scratch_regions, StorageClass::scratch}, {kv_regions, StorageClass::kv}}) {
        for (const auto& entry : entries)
            result.tensors.at(entry.at("tensor").get<size_t>()) = {
                storage_id, entry.at("offset").get<unsigned>() /
                                (storage_id == StorageClass::weight ? weight_element_bytes : 4)};
    }
    for (const auto& wave : expected_waves) {
        result.worker_loads.push_back(wave.at("estimated_work").get<std::vector<std::int64_t>>());
        for (const auto& lane : wave.at("workers")) {
            const auto begin = result.tasks.size();
            for (const auto& id : lane)
                result.tasks.push_back(encode(expected_tasks[id.get<int>()]));
            result.ranges.push_back(
                {static_cast<unsigned>(begin), static_cast<unsigned>(result.tasks.size() - begin)});
        }
    }
    return result;
}

} // namespace tileinfer
