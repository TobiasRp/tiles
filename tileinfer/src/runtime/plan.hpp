#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace tileinfer {

// Identifies a vocabulary entry with the same representation on host and device.
using TokenId = std::int32_t;

inline constexpr int kThreads = 1024;
inline constexpr const char* kAbi = "tileinfer-decoder-v4";
inline constexpr const char* kFingerprint = "decoder-wave-abi4-bf16-eager-simt-mma2-v1-sm86-b1024";
inline constexpr const char* kVersion3Fingerprint =
    "decoder-wave-abi3-warp-gemv-head-attention-rope-table-v1-sm86-b1024";
inline constexpr const char* kLegacyAbi = "tileinfer-decoder-v2";
inline constexpr const char* kLegacyFingerprint = "decoder-wave-abi2-ops-v3-stack-sm86-b128";
inline constexpr const char* kLegacyProbeFingerprint = "decoder-wave-abi2-probe-v1-sm86-b128";

// Storage precision for weights, activations and K/V; reductions use float32.
enum class WeightDtype : std::uint32_t
{
    float32,
    bfloat16,
};

// Names the validated arithmetic operations encoded in a decoder task.
enum class Opcode : std::uint32_t
{
    rms_norm_f32 = 1,
    gemv_f32 = 2,
    rope_query_f32 = 3,
    rope_cache_f32 = 4,
    attention_prefix_f32 = 5,
    silu_product_f32 = 6,
    rms_norm_bf16 = 7,
    gemv_bf16 = 8,
    rope_query_bf16 = 9,
    rope_cache_bf16 = 10,
    attention_prefix_bf16 = 11,
    silu_product_bf16 = 12,
};

// Chooses the buffer whose element offsets a tensor record indexes.
enum class StorageClass : std::uint32_t
{
    input = 0,
    output = 1,
    scratch = 2,
    kv = 3,
    weight = 4,
};

// Keeps the fixed block tensor layout shared by plans and the worker.
enum class TensorId : std::uint32_t
{
    x = 0,
    attn_norm = 1,
    u = 2,
    Wq = 3,
    q = 4,
    Wk = 5,
    k = 6,
    Wv = 7,
    v = 8,
    q_rot = 9,
    K = 10,
    V = 11,
    attention = 12,
    Wo = 13,
    r = 14,
    ffn_norm = 15,
    z = 16,
    Wgate = 17,
    gate = 18,
    Wup = 19,
    up = 20,
    product = 21,
    Wdown = 22,
    y = 23,
    count,
    none = std::numeric_limits<std::uint32_t>::max(),
};

inline constexpr std::size_t kTensorCount = static_cast<std::size_t>(TensorId::count);
inline constexpr std::uint32_t kNoTensor = std::numeric_limits<std::uint32_t>::max();

constexpr std::uint32_t tensor_index(TensorId id)
{
    return static_cast<std::uint32_t>(id);
}

// Compact operands and owned output region for one scheduled GPU task.
struct TaskRecord
{
    Opcode opcode;
    std::uint32_t a;
    std::uint32_t b;
    std::uint32_t c;
    std::uint32_t d;
    std::uint32_t e;
    std::uint32_t begin;
    std::uint32_t end;
    std::uint32_t head;
    std::uint32_t kv_head;
};

// Resolves one tensor ID to a storage buffer and element offset.
struct TensorRecord
{
    StorageClass storage;
    std::uint32_t offset; // elements of the storage class's dtype
};

// Carries the model's normalization and rotary-position constants.
struct NumericConfig
{
    float rms_norm_eps;
    float rope_theta;
    float rope_factor;
    float rope_low_freq_factor;
    float rope_high_freq_factor;
    float rope_original_context;
};

// Locates repeated layer weights, cache regions, and output stages.
struct StackConfig
{
    int layers;
    int vocab_size;
    int embedding_offset; // weight elements
    int layer_weights_offset;
    int layer_weight_stride;
    int final_norm_offset;
    int lm_head_offset;
    int layer_kv_stride;
    bool full_stack;
};

// Gives one logical worker its contiguous task slice for a wave.
struct WorkerTaskRange
{
    std::uint32_t begin;
    std::uint32_t count;
};

struct WorkerVariant
{
    std::string name;
    int registers_per_thread;
    int static_shared_bytes;
    int active_blocks_per_sm;
};

// Captures device and compiled-worker properties used to validate a plan.
struct Target
{
    std::string name;
    int major;
    int minor;
    int sm_count;
    bool cooperative_launch;
    int active_blocks_per_sm;
    int registers_per_thread;
    int static_shared_bytes;
    int max_threads_per_sm;
    std::uint64_t global_memory_bytes;
    int driver_version;
    int runtime_version;
    std::vector<WorkerVariant> variants;
};

// Host package data. Populate with load_plan() before uploading; callers are trusted.
struct LoadedPlan
{
    WeightDtype weight_dtype;
    int hidden_size;
    int intermediate_size;
    int query_heads;
    int kv_heads;
    int head_dim;
    int max_context;
    int workers;
    NumericConfig numeric;
    StackConfig stack;
    std::vector<TensorRecord> tensors;
    std::vector<float> rotary_frequencies;
    int wave_count;
    std::uint64_t scratch_bytes;
    std::uint64_t kv_bytes;
    std::uint64_t attention_bytes = 0;
    bool matrix_mma = false;
    std::vector<std::uint8_t> weights;
    std::vector<TaskRecord> tasks;
    std::vector<WorkerTaskRange> ranges;
    std::vector<std::vector<std::int64_t>> worker_loads;
    std::string plan_sha256;
    std::string weights_sha256;
};

inline constexpr int element_bytes(WeightDtype dtype)
{
    return dtype == WeightDtype::bfloat16 ? 2 : 4;
}

Target describe_target();
LoadedPlan load_plan(const std::string& package, const Target& target);
std::string sha256(const std::vector<std::uint8_t>& bytes);

} // namespace tileinfer
