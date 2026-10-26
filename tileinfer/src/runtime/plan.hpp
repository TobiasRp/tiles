#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace tileinfer {

// Identifies a vocabulary entry with the same representation on host and device.
using TokenId = std::int32_t;

inline constexpr int kThreads = 1024;
inline constexpr const char* kAbi = "tileinfer-decoder-v9";
inline constexpr const char* kFingerprint =
    "decoder-fused-abi9-bf16-eager-simt-wide-attention-v2-split-ffn-sm86-b1024";

// Shared-memory limits of the compiled worker; the loader rejects larger models.
inline constexpr int kMaxHidden = 4096; // a task's input vector lives in shared memory
inline constexpr int kMaxHeadDim = 128; // one K/V row per warp at most
inline constexpr int kMaxGroup = 8;     // query heads per K/V head, a power of two
// Shared bytes for one attention task's scores or probabilities of all its query heads.
inline constexpr int kAttentionSharedBytes = kMaxHidden * 4;

// Storage precision for weights, activations and K/V; reductions use float32.
enum class WeightDtype : std::uint32_t
{
    float32,
    bfloat16,
};

// Names one validated worker instruction; the worker's storage type selects precision.
enum class Opcode : std::uint32_t
{
    barrier = 0,           // grid barrier at the same stream position in every worker
    qkv = 1,               // [Q; K; V] rows: norm, project, Q and K unrotated, V cache write
    attention_scores = 2,  // one K/V head x context chunk: RoPE, scores, chunk softmax statistics
    attention_values = 3,  // one K/V head x context chunk: probabilities and partial P*V
    output_projection = 4, // Wo rows: combine attention partials, project, add the residual
    ffn = 5,               // product chunk: norm, gate and up rows, SiLU(gate) * up, down partial
    ffn_reduce = 6,        // y rows: sum the FFN chunks' down partials, add the residual
    embedding = 7,         // copies [begin,end) of the token's embedding row into hidden state
    final_rms_norm = 8,    // normalizes [begin,end) of the last layer's output
    logits = 9,            // projects vocabulary rows [begin,end)
};

// Package opcode names, without the _f32/_bf16 precision suffix.
inline constexpr std::array<const char*, 9> kOpcodeNames{"qkv", "attention_scores",
    "attention_values", "output_projection", "ffn", "ffn_reduce", "embedding", "final_rms_norm",
    "logits"};

// Every package opcode name the compiled worker accepts, for the target probe.
inline std::vector<std::string> supported_opcodes()
{
    std::vector<std::string> result;
    for (const char* suffix : {"_f32", "_bf16"})
        for (const char* name : kOpcodeNames)
            result.push_back(std::string(name) + suffix);
    return result;
}

// Chooses the buffer whose element offsets a tensor record indexes.
enum class StorageClass : std::uint32_t
{
    input = 0,
    output = 1,
    scratch = 2,
    kv = 3,
    weight = 4,
};

// Names the block tensors shared by plans and the worker; weights keep this order in a layer.
enum class TensorId : std::uint32_t
{
    x = 0,
    attn_norm = 1,
    Wq = 2,
    Wk = 3,
    Wv = 4,
    q = 5,
    k = 6,
    q_rot = 7,
    K = 8,
    V = 9,
    attention = 10,
    Wo = 11,
    r = 12,
    ffn_norm = 13,
    Wgate = 14,
    Wup = 15,
    product = 16,
    Wdown = 17,
    y = 18,
    count,
};

inline constexpr std::size_t kTensorCount = static_cast<std::size_t>(TensorId::count);
// Marks instructions that belong to no decoder layer: barriers and stack stages.
inline constexpr std::uint32_t kNoLayer = std::numeric_limits<std::uint32_t>::max();

constexpr std::uint32_t tensor_index(TensorId id)
{
    return static_cast<std::uint32_t>(id);
}

// One worker instruction. Rows index the opcode's output: the concatenated
// [Q; K; V] projection for qkv, the vocabulary for logits, and so on. A
// barrier with begin > 0 is skipped while the visible prefix has at most
// begin time steps; every worker sees the same position, so all skip it.
struct TaskRecord
{
    Opcode opcode;
    std::uint32_t layer; // decoder layer whose weights, K/V and hidden buffers apply
    std::uint32_t begin;
    std::uint32_t end;
    std::uint32_t kv_head; // attention tasks
    std::uint32_t chunk;   // attention tasks: context chunk; ffn tasks: product chunk
};

// Resolves one tensor ID to a storage buffer and element offset.
struct TensorRecord
{
    StorageClass storage;
    std::uint32_t offset; // elements of the storage class's dtype
};

// Element offsets of a layer's tensors: scratch is shared by all layers,
// K/V offsets are within one layer's cache, weights within one layer's block.
struct LayerLayout
{
    std::uint32_t q;
    std::uint32_t k;
    std::uint32_t q_rot;
    std::uint32_t attention;
    std::uint32_t r;
    std::uint32_t product;
    std::uint32_t key;
    std::uint32_t value;
    std::uint32_t attn_norm;
    std::uint32_t Wq;
    std::uint32_t Wk;
    std::uint32_t Wv;
    std::uint32_t Wo;
    std::uint32_t ffn_norm;
    std::uint32_t Wgate;
    std::uint32_t Wup;
    std::uint32_t Wdown;
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

// Gives one logical worker its contiguous instruction stream for a whole token.
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
    int attention_chunks;      // chunk tasks per K/V head in attention
    int attention_chunk_steps; // most time steps one chunk task covers
    int attention_fused_steps; // longest prefix one fused task per K/V head handles
    int ffn_chunks;            // FFN tasks per layer, each with one FP32 down partial
    int workers;
    NumericConfig numeric;
    StackConfig stack;
    std::vector<TensorRecord> tensors; // host-side layout, also used by diagnostics
    LayerLayout layout;
    std::vector<float> rotary_frequencies;
    int wave_count;
    int task_count;
    std::uint64_t scratch_bytes;
    std::uint64_t kv_bytes;
    std::uint64_t attention_bytes;         // scores workspace, [query_heads, max_context]
    std::uint64_t attention_partial_bytes; // FP32 statistics and partial P*V per head and chunk
    std::uint64_t ffn_partial_bytes;       // FP32 down partials, [ffn_chunks, hidden_size]
    std::vector<std::uint8_t> weights;
    // Worker streams, concatenated; one barrier separates consecutive waves in every stream.
    std::vector<TaskRecord> instructions;
    std::vector<WorkerTaskRange> streams; // one per worker
    // Host-only plan task ID of each instruction; -1 marks a barrier.
    std::vector<std::int32_t> instruction_tasks;
    std::vector<std::vector<std::int64_t>> worker_loads; // [wave][worker]
    std::string plan_sha256;
    std::string weights_sha256; // computed, or the plan's declared value when not verified
    bool weights_verified;
};

inline constexpr int element_bytes(WeightDtype dtype)
{
    return dtype == WeightDtype::bfloat16 ? 2 : 4;
}

Target describe_target();
// verify_weights = false trusts the plan's declared checksum and checks only the
// weight size; for repeated measurement runs of a package that was verified once.
LoadedPlan load_plan(const std::string& package, const Target& target, bool verify_weights = true);
std::string sha256(const std::vector<std::uint8_t>& bytes);

} // namespace tileinfer
