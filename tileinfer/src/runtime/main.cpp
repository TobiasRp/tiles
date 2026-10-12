#include "runtime.hpp"

#include <nlohmann/json.hpp>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
using json = nlohmann::json;

json target_json(const tileinfer::Target& t)
{
    return {{"device",
                {{"name", t.name}, {"compute_capability", {t.major, t.minor}},
                    {"sm_count", t.sm_count}, {"cooperative_launch", t.cooperative_launch},
                    {"global_memory_bytes", t.global_memory_bytes},
                    {"max_threads_per_sm", t.max_threads_per_sm},
                    {"driver_version", t.driver_version}, {"runtime_version", t.runtime_version}}},
        {"executor", {{"abi", tileinfer::kAbi},
                         {"opcodes", {"rms_norm_f32", "gemv_f32", "rope_query_f32",
                                         "rope_cache_f32", "attention_prefix_f32",
                                         "silu_product_f32", "embedding_lookup_f32", "logits_f32"}},
                         {"execution_ready", true}, {"threads_per_block", tileinfer::kThreads},
                         {"active_blocks_per_sm", t.active_blocks_per_sm},
                         {"registers_per_thread", t.registers_per_thread},
                         {"static_shared_bytes", t.static_shared_bytes},
                         {"fingerprint", tileinfer::kFingerprint}}}};
}

json inspect(const tileinfer::LoadedPlan& plan)
{
    return {{"plan_sha256", plan.plan_sha256}, {"weights_sha256", plan.weights_sha256},
        {"model",
            {{"hidden_size", plan.hidden_size}, {"intermediate_size", plan.intermediate_size},
                {"num_attention_heads", plan.query_heads}, {"num_key_value_heads", plan.kv_heads},
                {"head_dim", plan.head_dim}, {"max_context", plan.max_context}}},
        {"workers", plan.workers}, {"wave_count", plan.wave_count},
        {"task_count", plan.tasks.size()}, {"worker_loads", plan.worker_loads},
        {"scratch_bytes", plan.scratch_bytes}, {"kv_bytes", plan.kv_bytes},
        {"weight_bytes", plan.weights.size()},
        {"weight_dtype",
            plan.weight_dtype == tileinfer::WeightDtype::bfloat16 ? "bfloat16" : "float32"},
        {"model_kind",
            plan.stack.full_stack ? "llama_decoder_stack_f32" : "llama_decoder_block_f32"},
        {"layers", plan.stack.layers}, {"vocab_size", plan.stack.vocab_size}};
}

json run_block(const tileinfer::LoadedPlan& plan, const std::string& input_path)
{
    if (plan.stack.full_stack)
        throw std::runtime_error("expected a decoder block package");
    std::ifstream file(input_path);
    if (!file)
        throw std::runtime_error("cannot open block inputs");
    const json request = json::parse(file);
    tileinfer::SequenceContext context(plan);
    json steps = json::array();
    for (const auto& item : request.at("tokens")) {
        if (item.contains("reset") && item.at("reset") == true) {
            context.reset();
            continue;
        }
        const auto hidden = item.at("hidden").get<std::vector<float>>();
        const auto output = context.run_hidden(hidden);
        const auto scratch = context.scratch_snapshot();
        const auto kv = context.kv_snapshot();
        auto segment = [&](int tensor, int count) {
            const auto offset = plan.tensors.at(tensor).offset;
            return std::vector<float>(scratch.begin() + offset, scratch.begin() + offset + count);
        };
        const int kv_width = plan.kv_heads * plan.head_dim;
        const int p = context.position() - 1;
        const auto key_offset = plan.tensors.at(10).offset + p * kv_width;
        const auto value_offset = plan.tensors.at(11).offset + p * kv_width;
        const std::vector<float> key(kv.begin() + key_offset, kv.begin() + key_offset + kv_width);
        const std::vector<float> value(
            kv.begin() + value_offset, kv.begin() + value_offset + kv_width);
        auto residual = segment(14, plan.hidden_size);
        std::vector<float> ffn(plan.hidden_size);
        for (int i = 0; i < plan.hidden_size; ++i)
            ffn[i] = output[i] - residual[i];
        steps.push_back({{"position", p}, {"q", segment(9, plan.hidden_size)}, {"k", key},
            {"v", value}, {"attention", segment(12, plan.hidden_size)}, {"attn_residual", residual},
            {"ffn", ffn}, {"y", output}});
    }
    return {{"steps", steps}, {"position", context.position()}, {"valid", context.valid()}};
}

json run_stack(const tileinfer::LoadedPlan& plan, const std::string& input_path)
{
    if (!plan.stack.full_stack)
        throw std::runtime_error("expected a decoder stack package");
    std::ifstream file(input_path);
    if (!file)
        throw std::runtime_error("cannot open stack inputs");
    const json request = json::parse(file);
    tileinfer::SequenceContext context(plan);
    const bool trace_blocks = request.value("trace_blocks", false);
    if (trace_blocks)
        context.enable_block_trace();
    json steps = json::array();
    const int width = plan.kv_heads * plan.head_dim;
    for (const auto& item : request.at("tokens")) {
        if (item.is_object() && item.value("reset", false)) {
            context.reset();
            continue;
        }
        const int token_id = item.get<int>();
        const auto logits = context.run_token(token_id);
        const int position = context.position() - 1;
        const auto kv = context.kv_snapshot();
        json keys = json::array();
        json values = json::array();
        for (int layer = 0; layer < plan.stack.layers; ++layer) {
            const int base = layer * plan.stack.layer_kv_stride + position * width;
            const int key_offset = base + plan.tensors.at(10).offset;
            const int value_offset = base + plan.tensors.at(11).offset;
            keys.push_back(
                std::vector<float>(kv.begin() + key_offset, kv.begin() + key_offset + width));
            values.push_back(
                std::vector<float>(kv.begin() + value_offset, kv.begin() + value_offset + width));
        }
        json step = {{"token_id", token_id}, {"position", position}, {"logits", logits},
            {"k", keys}, {"v", values}};
        if (trace_blocks) {
            const auto hidden = context.block_trace_snapshot();
            json blocks = json::array();
            for (int layer = 0; layer < plan.stack.layers; ++layer) {
                const auto begin = hidden.begin() + layer * plan.hidden_size;
                blocks.push_back(std::vector<float>(begin, begin + plan.hidden_size));
            }
            step["blocks"] = std::move(blocks);
        }
        steps.push_back(std::move(step));
    }
    return {{"steps", steps}, {"position", context.position()}, {"valid", context.valid()}};
}

void state_smoke(const tileinfer::LoadedPlan& plan)
{
    tileinfer::SequenceContext context(plan);
    if (context.position() != 0 || !context.valid())
        throw std::runtime_error("bad new context state");
    for (int position = 0; position < plan.max_context; ++position) {
        if (context.begin_token() != position)
            throw std::runtime_error("bad token position");
        context.complete_token(true);
    }
    try {
        context.begin_token();
        throw std::runtime_error("capacity check did not fail");
    } catch (const std::runtime_error& error) {
        if (std::string_view(error.what()) != "sequence context capacity reached")
            throw;
    }
    context.reset();
    context.begin_token();
    context.complete_token(false);
    if (context.valid())
        throw std::runtime_error("failed call did not invalidate context");
    context.reset();
    if (context.position() != 0 || !context.valid())
        throw std::runtime_error("reset failed");
}

} // namespace

int main(int argc, char** argv)
{
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--help") {
            std::cout << "tile_runtime --describe-target-json\n"
                         "tile_runtime --inspect-plan PACKAGE\n"
                         "tile_runtime --state-smoke PACKAGE\n"
                         "tile_runtime --run-block PACKAGE INPUTS.json\n"
                         "tile_runtime --run-stack PACKAGE INPUTS.json\n";
            return 0;
        }
        if (argc < 2)
            throw std::runtime_error("missing mode; use --help");
        const auto target = tileinfer::describe_target();
        const std::string_view mode = argv[1];
        if (mode == "--describe-target-json" && argc == 2) {
            std::cout << target_json(target).dump(2) << '\n';
        } else if (mode == "--inspect-plan" && argc == 3) {
            std::cout << inspect(tileinfer::load_plan(argv[2], target)).dump(2) << '\n';
        } else if (mode == "--run-stack" && argc == 4) {
            std::cout << run_stack(tileinfer::load_plan(argv[2], target), argv[3]).dump() << '\n';
        } else if (mode == "--run-block" && argc == 4) {
            std::cout << run_block(tileinfer::load_plan(argv[2], target), argv[3]).dump() << '\n';
        } else if (mode == "--state-smoke" && argc == 3) {
            const auto plan = tileinfer::load_plan(argv[2], target);
            state_smoke(plan);
            std::cout << "{\"state_smoke\": true}\n";
        } else {
            throw std::runtime_error("unknown mode; use --help");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "tile_runtime: " << error.what() << '\n';
        return 2;
    }
}
