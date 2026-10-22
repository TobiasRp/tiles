#include "ops.cuh"

#include <cuda_bf16.h>
#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/launch>
#include <cuda/memory_pool>
#include <cuda/stream>
#include <nlohmann/json.hpp>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace tileinfer;
using json = nlohmann::json;

template <typename Scalar>
__global__ __launch_bounds__(kThreads, 1) void run_operator(int op, TaskRecord task,
    ops::BlockShape shape, const TensorRecord* records, Scalar* input, Scalar* output, Scalar* aux,
    Scalar* kv, const Scalar* weights, Scalar* scores, float* logits, int rows)
{
    __shared__ ops::SharedStorage shared;
    const ops::Context<Scalar> context{
        records, {input, output, aux, kv, weights, scores, nullptr}, shape, &shared};
    switch (op) {
    case 0:
        ops::gemv(task, context);
        break;
    case 1:
        ops::rms_norm(task, context);
        break;
    case 2:
        ops::rope_query(task, context);
        break;
    case 3:
        if constexpr (cuda::std::is_same_v<Scalar, float>)
            ops::attention_prefix_f32(task, context);
        else
            ops::attention_prefix_bf16(task, context);
        break;
    case 4:
        ops::silu_product(task, context);
        break;
    case 5:
        ops::embedding_lookup(0, shape.hidden_size, weights, output);
        break;
    case 6:
        ops::final_rms_norm(input, weights, output, shape.hidden_size, shape.numeric.rms_norm_eps,
            &shared.inverse_rms, shared.norm_partials);
        break;
    case 7:
        ops::logits(input, weights, logits, shape.hidden_size, rows, shared);
        break;
    }
}

template <typename Scalar>
std::vector<Scalar> values(const json& item, const char* name, std::size_t minimum)
{
    const auto supplied = item.value(name, std::vector<float>{});
    std::vector<Scalar> result(std::max(minimum, supplied.size()), static_cast<Scalar>(0.0f));
    for (std::size_t i = 0; i < supplied.size(); ++i)
        result[i] = static_cast<Scalar>(supplied[i]);
    return result;
}

template <typename Scalar>
json evaluate(const json& item)
{
    const std::string name = item.at("op");
    const std::vector<std::string> names{
        "gemv", "rms_norm", "rope", "attention", "silu", "embedding", "final_norm", "logits"};
    const auto found = std::find(names.begin(), names.end(), name);
    if (found == names.end())
        throw std::runtime_error("unknown operator");
    const int op = static_cast<int>(found - names.begin());
    const int width = item.value("width", 32);
    const int rows = item.value("rows", width);
    const int dh = item.value("head_dim", width);
    const int nq = item.value("heads", 1);
    const int nkv = item.value("kv_heads", 1);
    const int position = item.value("position", 0);
    const int capacity = item.value("capacity", position + 1);
    if (width <= 0 || rows <= 0 || dh <= 0 || capacity <= position || position < 0)
        throw std::runtime_error("invalid operator dimensions");
    const int output_count = op == 0 || op == 7 ? rows : op == 2 || op == 3 ? nq * dh : width;
    auto input = values<Scalar>(item, "input", std::max(width, nq * dh));
    auto weight = values<Scalar>(item, "weights", static_cast<std::size_t>(rows) * width);
    auto aux = values<Scalar>(item, "aux", std::max(width, rows));
    auto cache = values<Scalar>(item, "kv", 2 * capacity * nkv * dh);
    std::vector<Scalar> output(output_count, static_cast<Scalar>(-123.0f));
    std::vector<Scalar> scores(nq * capacity, static_cast<Scalar>(-123.0f));
    auto frequencies = item.value("frequencies", std::vector<float>(dh / 2, 0.01f));
    std::vector<TensorRecord> records(kTensorCount, {StorageClass::scratch, 0});
    records[0] = {StorageClass::input, 0};
    records[1] = {StorageClass::weight, 0};
    records[2] = {StorageClass::output, 0};
    records[3] = {StorageClass::scratch, 0};
    TaskRecord task{};
    task.a = 0;
    task.b = 1;
    task.c = 2;
    task.d = item.value("residual", false) ? 3 : kNoTensor;
    task.begin = item.value("begin", 0);
    task.end = item.value("end", output_count);
    task.head = item.value("head", 0);
    task.kv_head = item.value("kv_head", 0);
    if (op == 2)
        task.b = 2;
    if (op == 3) {
        records[1] = {StorageClass::kv, 0};
        records[2] = {StorageClass::kv, static_cast<std::uint32_t>(capacity * nkv * dh)};
        records[3] = {StorageClass::output, 0};
        task.b = 1;
        task.c = 2;
        task.d = 3;
    }
    if (op == 4)
        task.b = 3;
    cuda::device_ref device{0};
    cuda::stream stream{device};
    auto& pool = cuda::device_default_memory_pool(device);
    cuda::device_buffer<Scalar> d_input(stream, pool, input.size(), cuda::no_init);
    cuda::device_buffer<Scalar> d_weight(stream, pool, weight.size(), cuda::no_init);
    cuda::device_buffer<Scalar> d_aux(stream, pool, aux.size(), cuda::no_init);
    cuda::device_buffer<Scalar> d_cache(stream, pool, cache.size(), cuda::no_init);
    cuda::device_buffer<Scalar> d_output(stream, pool, output.size(), cuda::no_init);
    cuda::device_buffer<Scalar> d_scores(stream, pool, scores.size(), cuda::no_init);
    cuda::device_buffer<float> d_logits(stream, pool, output_count, cuda::no_init);
    cuda::device_buffer<float> d_frequencies(stream, pool, frequencies.size(), cuda::no_init);
    cuda::device_buffer<TensorRecord> d_records(stream, pool, records.size(), cuda::no_init);
    cuda::copy_bytes(stream, input, d_input);
    cuda::copy_bytes(stream, weight, d_weight);
    cuda::copy_bytes(stream, aux, d_aux);
    cuda::copy_bytes(stream, cache, d_cache);
    cuda::copy_bytes(stream, output, d_output);
    cuda::copy_bytes(stream, scores, d_scores);
    cuda::copy_bytes(stream, frequencies, d_frequencies);
    cuda::copy_bytes(stream, records, d_records);
    NumericConfig numeric{};
    numeric.rms_norm_eps = item.value("epsilon", 1e-5f);
    const ops::BlockShape shape{
        width, width, nkv, dh, position, capacity, numeric, d_frequencies.data()};
    const auto config = cuda::make_config(cuda::block_dims<kThreads>(), cuda::grid_dims(1));
    cuda::launch(stream, config, run_operator<Scalar>, op, task, shape, d_records.data(),
        d_input.data(), d_output.data(), d_aux.data(), d_cache.data(), d_weight.data(),
        d_scores.data(), d_logits.data(), rows);
    cuda::copy_bytes(stream, d_output, output);
    cuda::copy_bytes(stream, d_scores, scores);
    stream.sync();
    std::vector<float> result(output.size()), probabilities(scores.size());
    for (std::size_t i = 0; i < output.size(); ++i)
        result[i] = static_cast<float>(output[i]);
    for (std::size_t i = 0; i < scores.size(); ++i)
        probabilities[i] = static_cast<float>(scores[i]);
    if (op == 7) {
        cuda::copy_bytes(stream, d_logits, result);
        stream.sync();
    }
    return {{"output", result}, {"probabilities", probabilities}};
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
        for (const auto& item : cases) {
            if (item.at("dtype") == "bfloat16")
                results.push_back(evaluate<__nv_bfloat16>(item));
            else
                results.push_back(evaluate<float>(item));
        }
        std::cout << results.dump() << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
