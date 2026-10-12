#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tileinfer {

inline constexpr int kThreads = 128;
inline constexpr const char* kAbi = "tileinfer-mlp-v1";
inline constexpr const char* kFingerprint = "warp-gemv-cub-gelu-tanh-task-v1-sm86-b128";

enum class Opcode : std::uint32_t
{
    gemv_bias_gelu_f32 = 1,
    gemv_bias_f32 = 2
};
enum class Buffer : std::uint32_t
{
    input = 0,
    scratch = 1,
    output = 2
};

struct Task
{
    std::uint32_t opcode;
    std::uint32_t row_begin;
    std::uint32_t row_end;
    std::uint32_t width;
    std::uint32_t weight_offset; // float elements, relative to weights.bin
    std::uint32_t bias_offset;
    std::uint32_t input_buffer;
    std::uint32_t output_buffer;
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

struct Plan
{
    int d;
    int h;
    int o;
    int workers;
    int wave_count;
    std::vector<std::uint8_t> weights;
    std::vector<Task> tasks;
    std::vector<Range> ranges; // wave-major, then worker
    std::string plan_sha256;
    std::string weights_sha256;
};

Target describe_target();

class Executor
{
  public:
    explicit Executor(const Plan& plan);
    ~Executor();
    Executor(const Executor&) = delete;
    Executor& operator=(const Executor&) = delete;
    void run(const float* input, float* output, float* hidden = nullptr);
    void upload_input(const float* input);
    void launch();
    void synchronize();
    double timed_batch(int iterations);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

Plan load_plan(const std::string& package, const Target& target);
std::string sha256(const std::vector<std::uint8_t>& bytes);

} // namespace tileinfer
