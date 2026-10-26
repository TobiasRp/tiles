// Runs the decoder's device routines in isolation on one block.
//
//   ops_unit CASES.json
//
// Each case names an operator, its dimensions and any input buffers. Missing
// inputs are zero; outputs start at -123 so writes outside a task's rows show
// up. The result lists every buffer after the operator ran. Cases give Wdown
// as [hidden, intermediate]; it is stored transposed, as in a package.

#include "ops.cuh"

#include <cuda_bf16.h>
#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/launch>
#include <cuda/memory_pool>
#include <cuda/stream>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace tileinfer;
using json = nlohmann::json;

enum Operator
{
    qkv,
    attention,
    gate_up,
    down,
    ffn,
    embedding,
    final_norm,
    logits,
};

// Down projection rows [begin,end) through the FFN's split-K path: the down
// partial of each of shape.ffn_chunks even product chunks, then the
// reduction. With gate_up set, each chunk runs the whole FFN task.
template <typename Scalar>
__device__ void split_down(int begin, int end, bool with_gate_up, const ops::Layer<Scalar>& layer,
    const ops::Shape& shape, ops::SharedStorage& shared)
{
    const int f = shape.intermediate_size;
    for (int chunk = 0; chunk < shape.ffn_chunks; ++chunk) {
        const int first = f * chunk / shape.ffn_chunks;
        const int last = f * (chunk + 1) / shape.ffn_chunks;
        if (with_gate_up)
            ops::ffn(first, last, chunk, layer, shape, shared);
        else
            ops::down_partial(first, last, chunk, layer, shape, shared);
        __syncthreads();
    }
    ops::ffn_reduce(begin, end, layer, shape);
}

template <typename Scalar>
__global__ __launch_bounds__(kThreads, 1) void run_operator(int op, int begin, int end,
    ops::Layer<Scalar> layer, ops::Shape shape, const Scalar* weights, float* logits_out,
    Candidate* candidate)
{
    __shared__ ops::SharedStorage shared;
    switch (op) {
    case qkv:
        ops::qkv(begin, end, layer, shape, shared);
        break;
    case attention:
        // Every head's chunk tasks, then the output projection with Wo =
        // identity and a zero residual, which stores the combined attention
        // vector. Inactive chunk tasks and, for a fused prefix, values tasks
        // return at once, as in the worker.
        for (int kv_head = 0; kv_head < shape.kv_heads; ++kv_head)
            for (int chunk = 0; chunk < shape.attention_chunks; ++chunk) {
                ops::attention_scores(kv_head, chunk, layer, shape, shared);
                __syncthreads();
            }
        for (int kv_head = 0; kv_head < shape.kv_heads; ++kv_head)
            for (int chunk = 0; chunk < shape.attention_chunks; ++chunk) {
                ops::attention_values(kv_head, chunk, layer, shape, shared);
                __syncthreads();
            }
        ops::output_projection(0, shape.hidden_size, layer, shape, shared);
        break;
    case gate_up:
        ops::gate_up(begin, end, layer, shape, shared);
        break;
    case down:
        split_down(begin, end, false, layer, shape, shared);
        break;
    case ffn:
        split_down(begin, end, true, layer, shape, shared);
        break;
    case embedding:
        ops::embedding_lookup(0, begin, end, shape.hidden_size, weights, layer.y);
        break;
    case final_norm:
        ops::normalize(layer.x, weights, layer.y, begin, end, shape.hidden_size,
            shape.numeric.rms_norm_eps, shared);
        break;
    case logits:
        ops::logits(layer.x, weights, logits_out, candidate, begin, end, shape.hidden_size, shared);
        break;
    }
}

template <typename Scalar>
json evaluate(const json& item)
{
    const std::map<std::string, Operator> operators{{"qkv", qkv}, {"attention", attention},
        {"gate_up", gate_up}, {"down", down}, {"ffn", ffn}, {"embedding", embedding},
        {"final_norm", final_norm}, {"logits", logits}};
    const Operator op = operators.at(item.at("op").get<std::string>());
    const int d = item.value("hidden", 1);
    const int f = item.value("intermediate", 1);
    const int nq = item.value("heads", 1);
    const int nkv = item.value("kv_heads", 1);
    const int dh = item.value("head_dim", 2);
    const int position = item.value("position", 0);
    const int capacity = item.value("capacity", position + 1);
    // The planner's chunk rule for a `capacity`-step context (ops::active_chunks).
    const int chunk_steps = item.value("chunk_steps", capacity);
    const int fused_steps = item.value("fused_steps", chunk_steps);
    const int chunks =
        capacity <= fused_steps ? 1 : std::max(2, (capacity + chunk_steps - 1) / chunk_steps);
    const int rows = item.value("rows", d);
    const int kv_width = nkv * dh;
    const int ffn_chunks = std::min(f, item.value("ffn_chunks", 3));

    // Inputs default to zero, outputs to the sentinel; the case may supply either.
    std::map<std::string, std::vector<Scalar>> host;
    auto buffer = [&](const std::string& name, std::size_t count, float fill) {
        const auto supplied = item.value(name, std::vector<float>{});
        auto& values = host[name];
        values.assign(std::max(count, supplied.size()), static_cast<Scalar>(fill));
        for (std::size_t i = 0; i < supplied.size(); ++i)
            values[i] = static_cast<Scalar>(supplied[i]);
    };
    for (const char* name : {"x", "r", "q", "q_rot", "product"})
        buffer(name, std::max(d, f), 0.0f);
    buffer("k", kv_width, 0.0f);
    for (const char* name : {"key_cache", "value_cache"})
        buffer(name, static_cast<std::size_t>(capacity) * kv_width, 0.0f);
    for (const char* name : {"attn_norm", "ffn_norm"})
        buffer(name, d, 0.0f);
    buffer("Wq", static_cast<std::size_t>(d) * d, 0.0f);
    buffer("Wk", static_cast<std::size_t>(kv_width) * d, 0.0f);
    buffer("Wv", static_cast<std::size_t>(kv_width) * d, 0.0f);
    buffer("Wo", static_cast<std::size_t>(d) * d, 0.0f);
    buffer("Wgate", static_cast<std::size_t>(f) * d, 0.0f);
    buffer("Wup", static_cast<std::size_t>(f) * d, 0.0f);
    buffer("Wdown", static_cast<std::size_t>(d) * f, 0.0f);
    {
        // The package layout: [f, d].
        auto& down = host["Wdown"];
        std::vector<Scalar> transposed(down.size());
        for (int row = 0; row < d; ++row)
            for (int column = 0; column < f; ++column)
                transposed[static_cast<std::size_t>(column) * d + row] =
                    down[static_cast<std::size_t>(row) * f + column];
        down = std::move(transposed);
    }
    buffer("weights", static_cast<std::size_t>(rows) * d, 0.0f);
    if (item.contains("poison")) // JSON has no NaN or infinity
        host["weights"].at(static_cast<std::size_t>(20) * d + 5) = static_cast<Scalar>(
            item.at("poison") == "nan" ? std::numeric_limits<float>::quiet_NaN()
                                       : std::numeric_limits<float>::infinity());
    buffer("y", rows, -123.0f);
    buffer("attention", d, -123.0f);
    buffer("scores", static_cast<std::size_t>(nq) * capacity, -123.0f);
    // An operator's own output starts at the sentinel unless the case supplies it.
    const std::map<Operator, std::vector<const char*>> outputs{
        {qkv, {"q", "k"}}, {attention, {"r", "q_rot"}}, {gate_up, {"product"}}, {ffn, {"product"}}};
    if (outputs.contains(op))
        for (const char* name : outputs.at(op))
            if (!item.contains(name))
                std::fill(host[name].begin(), host[name].end(), static_cast<Scalar>(-123.0f));
    if (op == qkv && !item.contains("key_cache")) {
        std::fill(host["key_cache"].begin(), host["key_cache"].end(), static_cast<Scalar>(-123.0f));
        std::fill(
            host["value_cache"].begin(), host["value_cache"].end(), static_cast<Scalar>(-123.0f));
    }

    cuda::device_ref device{0};
    cuda::stream stream{device};
    auto& pool = cuda::device_default_memory_pool(device);
    std::map<std::string, cuda::device_buffer<Scalar>> buffers;
    for (const auto& [name, values] : host) {
        auto& target = buffers
                           .emplace(name, cuda::device_buffer<Scalar>(
                                              stream, pool, values.size(), cuda::no_init))
                           .first->second;
        cuda::copy_bytes(stream, values, target);
    }
    // Attention partials and statistics, then the FFN partials, as in a context.
    const std::size_t attention_partials = static_cast<std::size_t>(chunks) * nq * (dh + 2);
    const std::vector<float> zero_partials(
        attention_partials + static_cast<std::size_t>(ffn_chunks) * d, 0.0f);
    cuda::device_buffer<float> partials(stream, pool, zero_partials.size(), cuda::no_init);
    cuda::copy_bytes(stream, zero_partials, partials);
    const std::vector<float> sentinel(rows, -123.0f);
    cuda::device_buffer<float> d_logits(stream, pool, sentinel.size(), cuda::no_init);
    cuda::copy_bytes(stream, sentinel, d_logits);
    const std::vector<Candidate> no_candidate{{-123.0f, -123}};
    cuda::device_buffer<Candidate> d_candidate(stream, pool, 1, cuda::no_init);
    cuda::copy_bytes(stream, no_candidate, d_candidate);
    auto frequencies = item.value("frequencies", std::vector<float>(dh / 2, 0.01f));
    cuda::device_buffer<float> d_frequencies(stream, pool, frequencies.size(), cuda::no_init);
    cuda::copy_bytes(stream, frequencies, d_frequencies);

    auto at = [&](const char* name) { return buffers.at(name).data(); };
    const ops::Layer<Scalar> layer{at("x"), at("y"), at("q"), at("k"), at("q_rot"), at("attention"),
        at("r"), at("product"), at("key_cache"), at("value_cache"), at("scores"), partials.data(),
        partials.data() + static_cast<std::size_t>(chunks) * nq * dh,
        partials.data() + attention_partials, at("attn_norm"), at("Wq"), at("Wk"), at("Wv"),
        at("Wo"), at("ffn_norm"), at("Wgate"), at("Wup"), at("Wdown"), nullptr, nullptr, position};
    NumericConfig numeric{};
    numeric.rms_norm_eps = item.value("epsilon", 1e-5f);
    const ops::Shape shape{d, f, nq, nkv, dh, capacity, chunks, chunk_steps, fused_steps,
        ffn_chunks, numeric, d_frequencies.data()};
    const int begin = item.value("begin", 0);
    const int end = item.value("end", rows);
    const auto config = cuda::make_config(cuda::block_dims<kThreads>(), cuda::grid_dims(1));
    cuda::launch(stream, config, run_operator<Scalar>, static_cast<int>(op), begin, end, layer,
        shape, at("weights"), d_logits.data(), d_candidate.data());

    json result;
    for (auto& [name, values] : host) {
        // Only buffers an operator may write are returned; inputs stay on the device.
        static const std::set<std::string> written{"y", "r", "q", "k", "q_rot", "attention",
            "product", "key_cache", "value_cache", "scores"};
        if (!written.contains(name))
            continue;
        cuda::copy_bytes(stream, buffers.at(name), values);
        stream.sync();
        std::vector<float> widened(values.size());
        std::transform(values.begin(), values.end(), widened.begin(),
            [](Scalar value) { return static_cast<float>(value); });
        result[name] = widened;
    }
    std::vector<float> host_logits(sentinel.size());
    cuda::copy_bytes(stream, d_logits, host_logits);
    stream.sync();
    result["logits"] = host_logits;
    std::vector<Candidate> candidate(1);
    cuda::copy_bytes(stream, d_candidate, candidate);
    stream.sync();
    // A NaN value is not valid JSON; the candidate's ID marks it.
    result["candidate"] = {
        candidate[0].id == kNonFinite ? 0.0f : candidate[0].value, candidate[0].id};
    return result;
}
} // namespace

int main(int argc, char** argv)
{
    try {
        if (argc != 2)
            throw std::runtime_error("usage: ops_unit CASES.json");
        std::ifstream source(argv[1]);
        const auto cases = json::parse(source);
        json results = json::array();
        for (const auto& item : cases)
            results.push_back(item.at("dtype") == "bfloat16" ? evaluate<__nv_bfloat16>(item)
                                                             : evaluate<float>(item));
        std::cout << results.dump() << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
