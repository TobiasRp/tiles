#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tileinfer {

inline constexpr int kThreads = 128;
inline constexpr const char* kAbi = "tileinfer-decoder-v3";
inline constexpr const char* kFingerprint = "decoder-wave-abi3-weight-template-trace-sm86-b128";
inline constexpr const char* kLegacyAbi = "tileinfer-decoder-v2";
inline constexpr const char* kLegacyFingerprint = "decoder-wave-abi2-ops-v3-stack-sm86-b128";
inline constexpr const char* kLegacyProbeFingerprint = "decoder-wave-abi2-probe-v1-sm86-b128";

enum class WeightDtype : std::uint32_t
{
    float32,
    bfloat16,
};

enum class Opcode : std::uint32_t
{
    rms_norm_f32 = 1,
    gemv_f32 = 2,
    rope_query_f32 = 3,
    rope_cache_f32 = 4,
    attention_prefix_f32 = 5,
    silu_product_f32 = 6,
};

enum class StorageClass : std::uint32_t
{
    input = 0,
    output = 1,
    scratch = 2,
    kv = 3,
    weight = 4,
};

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
};

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

struct TensorRecord
{
    StorageClass storage;
    std::uint32_t offset; // elements of the storage class's dtype
};

struct NumericConfig
{
    float rms_norm_eps;
    float rope_theta;
    float rope_factor;
    float rope_low_freq_factor;
    float rope_high_freq_factor;
    float rope_original_context;
};

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

struct Range
{
    std::uint32_t begin;
    std::uint32_t count;
};

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
};

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
    int wave_count;
    std::uint64_t scratch_bytes;
    std::uint64_t kv_bytes;
    std::vector<std::uint8_t> weights;
    std::vector<TaskRecord> tasks;
    std::vector<Range> ranges;
    std::vector<std::vector<std::int64_t>> worker_loads;
    std::string plan_sha256;
    std::string weights_sha256;
};

struct BenchmarkResult
{
    std::vector<double> samples_us;
    int top_token;
    float top_logit;
};

Target describe_target();
LoadedPlan load_plan(const std::string& package, const Target& target);
std::string sha256(const std::vector<std::uint8_t>& bytes);

// Owns mutable storage and a stream. A failed call makes the cache unusable
// until reset; reset changes the valid prefix, without clearing old bytes.
class SequenceContext
{
  public:
    explicit SequenceContext(const LoadedPlan& plan);
    ~SequenceContext();
    SequenceContext(const SequenceContext&) = delete;
    SequenceContext& operator=(const SequenceContext&) = delete;

    int begin_token();
    void complete_token(bool success);
    void reset();
    [[nodiscard]] int position() const;
    [[nodiscard]] bool valid() const;
    std::vector<float> run_hidden(const std::vector<float>& hidden);
    std::vector<float> run_token(int token_id);
    BenchmarkResult benchmark_token(
        int position, int token_id, int warmup, int iterations, int rounds, bool profile = false);
    void enable_block_trace();
    [[nodiscard]] std::vector<float> block_trace_snapshot() const;
    [[nodiscard]] std::vector<float> scratch_snapshot() const;
    [[nodiscard]] std::vector<float> kv_snapshot() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tileinfer
