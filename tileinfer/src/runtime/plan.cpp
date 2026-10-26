#include "plan.hpp"
#include "detail/plan_validation.hpp"

#include <picosha2.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numbers>
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

std::uint64_t multiply(std::uint64_t a, std::uint64_t b)
{
    require(b == 0 || a <= std::numeric_limits<std::uint64_t>::max() / b, "storage size overflow");
    return a * b;
}

std::uint64_t add(std::uint64_t a, std::uint64_t b)
{
    require(a <= std::numeric_limits<std::uint64_t>::max() - b, "storage size overflow");
    return a + b;
}

std::uint64_t align64(std::uint64_t value)
{
    return add(value, (64 - value % 64) % 64);
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
    if (value.is_number_unsigned())
        require(value.get<std::uint64_t>() <=
                    static_cast<std::uint64_t>(std::numeric_limits<int>::max()),
            std::string("invalid ") + name);
    const auto n = value.get<std::int64_t>();
    require(n > 0 && n <= std::numeric_limits<int>::max(), std::string("invalid ") + name);
    return static_cast<int>(n);
}

float positive_float(const json& value, const char* name)
{
    require(value.is_number(), std::string("invalid ") + name);
    const double n = value.get<double>();
    require(std::isfinite(n) && n > 0 && n <= std::numeric_limits<float>::max(),
        std::string("invalid ") + name);
    const float result = static_cast<float>(n);
    require(result > 0, std::string("invalid ") + name);
    return result;
}

using Spec = detail::PlanShape;

constexpr std::array<const char*, kTensorCount> names{"x", "attn_norm", "Wq", "Wk", "Wv", "q", "k",
    "q_rot", "K", "V", "attention", "Wo", "r", "ffn_norm", "Wgate", "Wup", "product", "Wdown", "y"};
using Id = TensorId;
constexpr std::array weight_ids{
    Id::attn_norm, Id::Wq, Id::Wk, Id::Wv, Id::Wo, Id::ffn_norm, Id::Wgate, Id::Wup, Id::Wdown};
constexpr std::array scratch_ids{Id::q, Id::k, Id::q_rot, Id::attention, Id::r, Id::product};
constexpr std::array kv_ids{Id::K, Id::V};

std::vector<int> shape(const Spec& s, Id id)
{
    const int kv_width = s.nkv * s.dh;
    switch (id) {
    case Id::attn_norm:
    case Id::ffn_norm:
        return {s.d};
    case Id::Wq:
    case Id::Wo:
        return {s.d, s.d};
    case Id::Wk:
    case Id::Wv:
        return {kv_width, s.d};
    case Id::k:
        return {kv_width};
    case Id::Wgate:
    case Id::Wup:
        return {s.f, s.d};
    case Id::Wdown:
        return {s.d, s.f};
    case Id::product:
        return {s.f};
    case Id::K:
    case Id::V:
        return {1, s.context, s.nkv, s.dh};
    default:
        return {s.d};
    }
}

std::string storage_class(Id id)
{
    if (std::find(weight_ids.begin(), weight_ids.end(), id) != weight_ids.end())
        return "weight";
    if (id == Id::K || id == Id::V)
        return "kv";
    if (id == Id::x)
        return "input";
    if (id == Id::y)
        return "output";
    return "scratch";
}

json descriptors(const Spec& s, std::string_view weight_dtype)
{
    json result = json::array();
    for (int id = 0; id < static_cast<int>(names.size()); ++id) {
        const auto dims = shape(s, static_cast<Id>(id));
        std::vector<int> strides(dims.size(), 1);
        for (int index = static_cast<int>(dims.size()) - 2; index >= 0; --index)
            strides[index] = strides[index + 1] * dims[index + 1];
        result.push_back({{"id", id}, {"name", names[id]}, {"shape", dims}, {"strides", strides},
            {"dtype", weight_dtype}, {"storage", storage_class(static_cast<Id>(id))}});
    }
    return result;
}

template <std::size_t N>
std::pair<json, std::uint64_t> regions(
    const Spec& s, const std::array<Id, N>& ids, int alignment, int element_bytes = 4)
{
    json result = json::array();
    std::uint64_t cursor = 0;
    for (Id id : ids) {
        cursor += (alignment - cursor % alignment) % alignment;
        std::uint64_t bytes = element_bytes;
        for (int dimension : shape(s, id))
            bytes = multiply(bytes, static_cast<std::uint64_t>(dimension));
        result.push_back({{"tensor", tensor_index(id)}, {"offset", cursor}, {"bytes", bytes},
            {"alignment", alignment}});
        cursor = add(cursor, bytes);
    }
    return {result, cursor};
}

} // namespace

std::string sha256(const std::vector<std::uint8_t>& bytes)
{
    return picosha2::hash256_hex_string(bytes);
}

LoadedPlan load_plan(const std::string& package, const Target& device, bool verify_weights)
{
    const auto plan_data = read_bytes(std::filesystem::path(package) / "plan.json");
    auto weights = read_bytes(std::filesystem::path(package) / "weights.bin");
    const auto document = json::parse(plan_data.begin(), plan_data.end());
    const auto& header = document.at("header");
    const auto model_kind = header.at("model_kind").get<std::string>();
    const bool full_stack = model_kind == "llama_decoder_stack";
    const int version = positive(header.at("format_version"), "format_version");
    require(version == 8 && header.at("executor_abi") == kAbi &&
                model_kind == (full_stack ? "llama_decoder_stack" : "llama_decoder_block"),
        "unsupported decoder plan header; regenerate the package with the current planner");
    const auto& model = document.at("model");
    const std::string weight_dtype = model.at("weight_dtype").get<std::string>();
    require(weight_dtype == "float32" || weight_dtype == "bfloat16", "unsupported weight dtype");
    const bool bf16 = weight_dtype == "bfloat16";
    const int weight_element_bytes = bf16 ? 2 : 4;
    require(!model.contains("dtype") && !model.contains("matrix_kernel") &&
                model.at("activation_dtype") == weight_dtype &&
                model.at("kv_dtype") == weight_dtype &&
                model.at("accumulation_dtype") == "float32" &&
                model.at("logits_compute_dtype") == weight_dtype &&
                model.at("api_output_dtype") == "float32" &&
                model.at("numeric_policy") == (bf16 ? "hf-llama-eager-bf16-v1" : "decoder-f32-v1"),
        "inconsistent numeric policy");
    Spec s{positive(model.at("hidden_size"), "hidden_size"),
        positive(model.at("intermediate_size"), "intermediate_size"),
        positive(model.at("num_attention_heads"), "num_attention_heads"),
        positive(model.at("num_key_value_heads"), "num_key_value_heads"),
        positive(model.at("head_dim"), "head_dim"),
        positive(model.at("max_context"), "max_context"),
        positive(model.at("attention_chunks"), "attention_chunks"),
        positive(model.at("attention_chunk_steps"), "attention_chunk_steps"),
        positive(model.at("attention_fused_steps"), "attention_fused_steps")};
    const int layers = positive(model.at("layers"), "layers");
    const int vocab = full_stack ? positive(model.at("vocab_size"), "vocab_size") : 0;
    const bool tied = full_stack ? model.at("tie_word_embeddings").get<bool>() : false;
    require(s.d == static_cast<std::int64_t>(s.nq) * s.dh && s.nq % s.nkv == 0 && s.dh % 2 == 0 &&
                (full_stack || layers == 1),
        "unsupported decoder dimensions");
    require(s.d <= kMaxHidden && s.dh <= kMaxHeadDim && s.nq / s.nkv <= kMaxGroup,
        "decoder dimensions exceed the compiled worker's shared-memory limits");
    // Attention lanes load K/V rows as 16-byte vectors, a power-of-two number
    // of lanes per row, and give each query head of a group the same warps.
    const int group = s.nq / s.nkv;
    const int row_bytes = s.dh * weight_element_bytes;
    const auto power_of_two = [](int value) { return (value & (value - 1)) == 0; };
    require(power_of_two(group) && row_bytes % 16 == 0 && power_of_two(row_bytes / 16) &&
                row_bytes / 16 <= 32,
        "attention shape unsupported by the compiled worker");
    // Every chunk task keeps its heads' probabilities in shared memory, and
    // the program holds the chunk tasks of the longest prefix (see
    // ops::active_chunks: one up to the fused length, then at least two).
    const std::int64_t longest_chunks =
        s.context <= s.fused ? 1
                             : std::max<std::int64_t>(2,
                                   (static_cast<std::int64_t>(s.context) + s.steps - 1) / s.steps);
    require(static_cast<std::int64_t>(s.steps) * group * weight_element_bytes <=
                    kAttentionSharedBytes &&
                s.fused <= s.steps && s.chunks == longest_chunks,
        "invalid attention chunking");
    for (std::size_t id = 0; id < kTensorCount; ++id) {
        std::uint64_t elements = 1;
        for (int dimension : shape(s, static_cast<Id>(id)))
            elements = multiply(elements, static_cast<unsigned>(dimension));
        require(elements <= static_cast<std::uint64_t>(std::numeric_limits<int>::max()),
            "tensor exceeds device address range");
    }
    require(model.at("rope_type") == "llama3" && model.at("rope_pairing") == "split_half" &&
                model.at("kv_cache_layout") == "[layers,max_context,kv_heads,head_dim]",
        "unsupported decoder numeric semantics");
    const NumericConfig numeric{positive_float(model.at("rms_norm_eps"), "rms_norm_eps"),
        positive_float(model.at("rope_theta"), "rope_theta"),
        positive_float(model.at("rope_factor"), "rope_factor"),
        positive_float(model.at("rope_low_freq_factor"), "rope_low_freq_factor"),
        positive_float(model.at("rope_high_freq_factor"), "rope_high_freq_factor"),
        static_cast<float>(
            positive(model.at("rope_original_max_position_embeddings"), "rope original context"))};
    require(numeric.rope_high_freq_factor > numeric.rope_low_freq_factor,
        "unsupported decoder numeric semantics");
    json head_map = json::array();
    for (int head = 0; head < s.nq; ++head)
        head_map.push_back(head / (s.nq / s.nkv));
    require(
        model.at("head_to_kv") == head_map && model.at("tensors") == descriptors(s, weight_dtype),
        "invalid tensor descriptors or GQA mapping");

    const auto& target = document.at("target");
    const int workers = positive(target.at("worker_count"), "worker_count");
    const auto& fingerprint = target.at("kernel_fingerprint");
    require(target.at("compute_capability") == json::array({device.major, device.minor}) &&
                target.at("sm_count") == device.sm_count && workers == device.sm_count &&
                target.at("cooperative_launch") == true && device.cooperative_launch &&
                target.at("threads_per_block") == kThreads && fingerprint == kFingerprint &&
                target.at("attention_shared_bytes") == kAttentionSharedBytes &&
                workers <= static_cast<std::int64_t>(device.sm_count) * device.active_blocks_per_sm,
        "decoder plan target does not match compiled worker");

    const auto& storage = document.at("storage");
    const auto [weight_regions, weight_bytes] = regions(s, weight_ids, 64, weight_element_bytes);
    const auto [scratch_regions, scratch_bytes] = regions(s, scratch_ids, 16, weight_element_bytes);
    const auto [kv_regions, kv_bytes] = regions(s, kv_ids, 16, weight_element_bytes);
    StackConfig stack{layers, vocab, 0, 0, 0, 0, 0, 0, full_stack};
    const std::uint64_t expected_kv_bytes =
        full_stack ? multiply(kv_bytes, static_cast<std::uint64_t>(layers)) : kv_bytes;
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
        const auto norm_offset = align64(add(layer_offset, multiply(layers, layer_stride)));
        const auto lm_offset =
            tied ? 0 : align64(add(norm_offset, multiply(s.d, weight_element_bytes)));
        const auto total = tied ? add(norm_offset, multiply(s.d, weight_element_bytes))
                                : add(lm_offset, embedding_bytes);
        require(
            total <= static_cast<std::uint64_t>(std::numeric_limits<int>::max()) *
                         weight_element_bytes &&
                expected_kv_bytes <= static_cast<std::uint64_t>(std::numeric_limits<int>::max()) *
                                         weight_element_bytes,
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
        stack.embedding_offset = 0;
        stack.layer_weights_offset = static_cast<int>(layer_offset / weight_element_bytes);
        stack.layer_weight_stride = static_cast<int>(layer_stride / weight_element_bytes);
        stack.final_norm_offset = static_cast<int>(norm_offset / weight_element_bytes);
        stack.lm_head_offset = static_cast<int>(lm_offset / weight_element_bytes);
        stack.layer_kv_stride = static_cast<int>(kv_bytes / weight_element_bytes);
    } else {
        require(
            storage.at("input_bytes") == static_cast<std::uint64_t>(s.d) * weight_element_bytes &&
                storage.at("output_bytes") ==
                    static_cast<std::uint64_t>(s.d) * weight_element_bytes &&
                storage.at("weight_bytes") == weight_bytes && weight_bytes == weights.size(),
            "invalid decoder block storage");
    }
    require(weight_bytes / weight_element_bytes <=
                    static_cast<std::uint64_t>(std::numeric_limits<int>::max()) &&
                scratch_bytes / weight_element_bytes <=
                    static_cast<std::uint64_t>(std::numeric_limits<int>::max()) &&
                expected_kv_bytes / weight_element_bytes <=
                    static_cast<std::uint64_t>(std::numeric_limits<int>::max()),
        "decoder storage exceeds device address range");
    const auto& integrity = document.at("integrity");
    require(integrity.at("weights_bytes") == weights.size(), "decoder weight size mismatch");
    // Hashing the weights dominates the load time (about 19 s for Llama 3.2 1B).
    const auto weights_sha256 =
        verify_weights ? sha256(weights) : integrity.at("weights_sha256").get<std::string>();
    require(integrity.at("weights_sha256") == weights_sha256, "decoder weight checksum mismatch");

    LoadedPlan result{};
    const auto attention_bytes = multiply(multiply(s.nq, s.context), weight_element_bytes);
    const auto partial_bytes = multiply(multiply(multiply(s.chunks, s.nq), s.dh + 2), 4);
    require(storage.at("attention_bytes") == attention_bytes &&
                storage.at("attention_partial_bytes") == partial_bytes,
        "invalid attention workspace");
    require(attention_bytes / weight_element_bytes <=
                static_cast<std::uint64_t>(std::numeric_limits<int>::max()),
        "attention workspace exceeds address range");
    result.attention_bytes = attention_bytes;
    result.attention_partial_bytes = partial_bytes;
    result.attention_chunks = s.chunks;
    result.attention_chunk_steps = s.steps;
    result.attention_fused_steps = s.fused;
    result.weight_dtype = weight_dtype == "bfloat16" ? WeightDtype::bfloat16 : WeightDtype::float32;
    result.hidden_size = s.d;
    result.intermediate_size = s.f;
    result.query_heads = s.nq;
    result.kv_heads = s.nkv;
    result.head_dim = s.dh;
    result.max_context = s.context;
    result.workers = workers;
    result.stack = stack;
    result.numeric = numeric;
    detail::validate_program(document, s, workers, result);
    result.scratch_bytes = scratch_bytes;
    result.kv_bytes = expected_kv_bytes;
    result.weights = std::move(weights);
    result.plan_sha256 = sha256(plan_data);
    result.weights_sha256 = weights_sha256;
    result.weights_verified = verify_weights;
    result.tensors.resize(names.size());
    result.tensors[tensor_index(Id::x)] = {StorageClass::input, 0};
    result.tensors[tensor_index(Id::y)] = {StorageClass::output, 0};
    for (const auto& [entries, storage_id] : std::array<std::pair<json, StorageClass>, 3>{
             std::pair{weight_regions, StorageClass::weight},
             {scratch_regions, StorageClass::scratch}, {kv_regions, StorageClass::kv}}) {
        for (const auto& entry : entries)
            result.tensors.at(entry.at("tensor").get<size_t>()) = {
                storage_id, static_cast<std::uint32_t>(
                                entry.at("offset").get<std::uint64_t>() / weight_element_bytes)};
    }
    const auto offset = [&](Id id) { return result.tensors[tensor_index(id)].offset; };
    result.layout = {offset(Id::q), offset(Id::k), offset(Id::q_rot), offset(Id::attention),
        offset(Id::r), offset(Id::product), offset(Id::K), offset(Id::V), offset(Id::attn_norm),
        offset(Id::Wq), offset(Id::Wk), offset(Id::Wv), offset(Id::Wo), offset(Id::ffn_norm),
        offset(Id::Wgate), offset(Id::Wup), offset(Id::Wdown)};
    for (int pair = 0; pair < s.dh / 2; ++pair) {
        const float inverse = 1.0f / std::pow(numeric.rope_theta, 2.0f * pair / s.dh);
        const float wavelength = 2.0f * std::numbers::pi_v<float> / inverse;
        const float low = numeric.rope_original_context / numeric.rope_low_freq_factor;
        const float high = numeric.rope_original_context / numeric.rope_high_freq_factor;
        const float smooth =
            (numeric.rope_original_context / wavelength - numeric.rope_low_freq_factor) /
            (numeric.rope_high_freq_factor - numeric.rope_low_freq_factor);
        const float frequency = wavelength > low ? inverse / numeric.rope_factor
                                : wavelength < high
                                    ? inverse
                                    : inverse * ((1.0f - smooth) / numeric.rope_factor + smooth);
        require(std::isfinite(frequency) && frequency >= 0, "invalid derived rotary frequency");
        result.rotary_frequencies.push_back(frequency);
    }
    return result;
}

} // namespace tileinfer
