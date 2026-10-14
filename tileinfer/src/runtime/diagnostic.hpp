#pragma once

#include "runtime.hpp"

#include <memory>
#include <vector>

namespace tileinfer {

// Retains the legacy fixed-position profiler's samples and top logit.
struct BenchmarkResult
{
    std::vector<double> samples_us;
    TokenId top_token;
    float top_logit;
};

// Blocking intermediate-value adapter used by correctness tests only.
class DiagnosticContext
{
  public:
    explicit DiagnosticContext(const LoadedPlan& plan);
    ~DiagnosticContext();
    DiagnosticContext(const DiagnosticContext&) = delete;
    DiagnosticContext& operator=(const DiagnosticContext&) = delete;

    int begin_token();
    void complete_token(bool success);
    void reset();
    [[nodiscard]] int position() const;
    [[nodiscard]] bool valid() const;
    std::vector<float> run_hidden(const std::vector<float>& hidden);
    std::vector<float> run_token(TokenId token_id);
    BenchmarkResult benchmark_token(int position, TokenId token_id, int warmup, int iterations,
        int rounds, bool profile = false);
    void enable_block_trace();
    [[nodiscard]] std::vector<float> block_trace_snapshot() const;
    [[nodiscard]] std::vector<float> scratch_snapshot() const;
    [[nodiscard]] std::vector<float> kv_snapshot() const;
    [[nodiscard]] std::vector<float> ffn_snapshot() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tileinfer
