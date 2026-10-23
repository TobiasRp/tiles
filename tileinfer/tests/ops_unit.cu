// Runs the decoder's device routines in isolation on one block.
//
//   ops_unit CASES.json
//
// Each case names an operator, its dimensions and any input buffers. Missing
// inputs are zero; outputs start at -123 so writes outside a task's rows show
// up. The result lists every buffer after the operator ran.

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
    embedding,
    final_norm,
    logits,
};

template <typename Scalar>
__global__ __launch_bounds__(kThreads, 1) void run_operator(int op, int begin, int end,
    ops::Layer<Scalar> layer, ops::Shape shape, const Scalar* weights, float* logits_out)
{
    __shared__ ops::SharedStorage shared;
    switch (op) {
    case qkv:
        ops::qkv(begin, layer, shape, shared);
        break;
    case attention:
        // Every head's chunks, then the output projection with Wo = identity
        // and a zero residual, which stores the combined attention vector.
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
        ops::down(begin, end, layer, shape, shared);
        break;
    case embedding:
        ops::embedding_lookup(0, begin, end, shape.hidden_size, weights, layer.y);
        break;
    case final_norm:
        ops::normalize(layer.x, weights, layer.y, begin, end, shape.hidden_size,
            shape.numeric.rms_norm_eps, shared);
        break;
    case logits:
        ops::logits(layer.x, weights, logits_out, begin, end, shape.hidden_size, shared);
        break;
    }
}

template <typename Scalar>
json evaluate(const json& item)
{
    const std::map<std::string, Operator> operators{{"qkv", qkv}, {"attention", attention},
        {"gate_up", gate_up}, {"down", down}, {"embedding", embedding}, {"final_norm", final_norm},
        {"logits", logits}};
    const Operator op = operators.at(item.at("op").get<std::string>());
    const int d = item.value("hidden", 1);
    const int f = item.value("intermediate", 1);
    const int nq = item.value("heads", 1);
    const int nkv = item.value("kv_heads", 1);
    const int dh = item.value("head_dim", 2);
    const int position = item.value("position", 0);
    const int capacity = item.value("capacity", position + 1);
    const int chunks = item.value("chunks", 1);
    const int rows = item.value("rows", d);
    const int kv_width = nkv * dh;

    // Inputs default to zero, outputs to the sentinel; the case may supply either.
    std::map<std::string, std::vector<Scalar>> host;
    auto buffer = [&](const std::string& name, std::size_t count, float fill) {
        const auto supplied = item.value(name, std::vector<float>{});
        auto& values = host[name];
        values.assign(std::max(count, supplied.size()), static_cast<Scalar>(fill));
        for (std::size_t i = 0; i < supplied.size(); ++i)
            values[i] = static_cast<Scalar>(supplied[i]);
    };
    for (const char* name : {"x", "r", "q_rot", "product"})
        buffer(name, std::max(d, f), 0.0f);
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
    buffer("weights", static_cast<std::size_t>(rows) * d, 0.0f);
    buffer("y", rows, -123.0f);
    buffer("attention", d, -123.0f);
    buffer("scores", static_cast<std::size_t>(nq) * capacity, -123.0f);
    // An operator's own output starts at the sentinel unless the case supplies it.
    const std::map<Operator, std::vector<const char*>> outputs{
        {qkv, {"q_rot"}}, {attention, {"r"}}, {gate_up, {"product"}}};
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
    const std::vector<float> zero_partials(static_cast<std::size_t>(chunks) * nq * (dh + 2), 0.0f);
    cuda::device_buffer<float> partials(stream, pool, zero_partials.size(), cuda::no_init);
    cuda::copy_bytes(stream, zero_partials, partials);
    const std::vector<float> sentinel(rows, -123.0f);
    cuda::device_buffer<float> d_logits(stream, pool, sentinel.size(), cuda::no_init);
    cuda::copy_bytes(stream, sentinel, d_logits);
    auto frequencies = item.value("frequencies", std::vector<float>(dh / 2, 0.01f));
    cuda::device_buffer<float> d_frequencies(stream, pool, frequencies.size(), cuda::no_init);
    cuda::copy_bytes(stream, frequencies, d_frequencies);

    auto at = [&](const char* name) { return buffers.at(name).data(); };
    const ops::Layer<Scalar> layer{at("x"), at("y"), at("q_rot"), at("attention"), at("r"),
        at("product"), at("key_cache"), at("value_cache"), at("scores"), partials.data(),
        partials.data() + static_cast<std::size_t>(chunks) * nq * dh, at("attn_norm"), at("Wq"),
        at("Wk"), at("Wv"), at("Wo"), at("ffn_norm"), at("Wgate"), at("Wup"), at("Wdown"), nullptr,
        nullptr};
    NumericConfig numeric{};
    numeric.rms_norm_eps = item.value("epsilon", 1e-5f);
    const ops::Shape shape{
        d, f, nq, nkv, dh, position, capacity, chunks, numeric, d_frequencies.data()};
    const int begin = item.value("begin", 0);
    const int end = item.value("end", rows);
    const auto config = cuda::make_config(cuda::block_dims<kThreads>(), cuda::grid_dims(1));
    cuda::launch(stream, config, run_operator<Scalar>, static_cast<int>(op), begin, end, layer,
        shape, at("weights"), d_logits.data());

    json result;
    for (auto& [name, values] : host) {
        // Only buffers an operator may write are returned; inputs stay on the device.
        static const std::set<std::string> written{
            "y", "r", "q_rot", "attention", "product", "key_cache", "value_cache", "scores"};
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
