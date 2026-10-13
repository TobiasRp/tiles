#include "runtime.hpp"
#ifdef TILE_TEST_HARNESS
#include "diagnostic.hpp"
#endif

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
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

tileinfer::TokenId parse_token_id(const json& item)
{
    if (!item.is_number_integer())
        throw std::runtime_error("token ID must be an integer");
    if (item.is_number_unsigned()) {
        const auto value = item.get<std::uint64_t>();
        if (value > static_cast<std::uint64_t>(std::numeric_limits<tileinfer::TokenId>::max()))
            throw std::runtime_error("token ID exceeds TokenId range");
        return static_cast<tileinfer::TokenId>(value);
    }
    const auto value = item.get<std::int64_t>();
    if (value < std::numeric_limits<tileinfer::TokenId>::min() ||
        value > std::numeric_limits<tileinfer::TokenId>::max())
        throw std::runtime_error("token ID exceeds TokenId range");
    return static_cast<tileinfer::TokenId>(value);
}

#ifdef TILE_TEST_HARNESS
json run_block(const tileinfer::LoadedPlan& plan, const std::string& input_path)
{
    if (plan.stack.full_stack)
        throw std::runtime_error("expected a decoder block package");
    std::ifstream file(input_path);
    if (!file)
        throw std::runtime_error("cannot open block inputs");
    const json request = json::parse(file);
    tileinfer::DiagnosticContext context(plan);
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
#endif

json run_stack(
    const tileinfer::LoadedPlan& plan, const std::string& input_path, bool trace_override = false)
{
    if (!plan.stack.full_stack)
        throw std::runtime_error("expected a decoder stack package");
    std::ifstream file(input_path);
    if (!file)
        throw std::runtime_error("cannot open stack inputs");
    const json request = json::parse(file);
    tileinfer::DecoderModel model(plan);
    auto context = model.create_context();
    const bool trace_blocks = trace_override || request.value("trace_blocks", false);
    json steps = json::array();
    const int width = plan.kv_heads * plan.head_dim;
    int final_position = 0;
    for (const auto& item : request.at("tokens")) {
        if (item.is_object() && item.value("reset", false)) {
            auto reset = context.enqueue_reset();
            final_position = context.wait(reset).position;
            continue;
        }
        const tileinfer::TokenId token_id = parse_token_id(item);
        auto ticket = context.enqueue_decode(token_id, {true, true, trace_blocks});
        const auto completion = context.wait(ticket);
        const auto snapshot = context.collect(ticket);
        const int position = completion.position - 1;
        final_position = completion.position;
        const auto& kv = snapshot.kv;
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
        json step = {{"token_id", token_id}, {"position", position}, {"logits", snapshot.logits},
            {"k", keys}, {"v", values}};
        if (trace_blocks) {
            const auto& hidden = snapshot.block_outputs;
            json blocks = json::array();
            for (int layer = 0; layer < plan.stack.layers; ++layer) {
                const auto begin = hidden.begin() + layer * plan.hidden_size;
                blocks.push_back(std::vector<float>(begin, begin + plan.hidden_size));
            }
            step["blocks"] = std::move(blocks);
        }
        steps.push_back(std::move(step));
    }
    return {{"steps", steps}, {"position", final_position}, {"valid", true}};
}

#ifdef TILE_TEST_HARNESS
void state_smoke(const tileinfer::LoadedPlan& plan)
{
    tileinfer::DiagnosticContext context(plan);
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
#endif

std::vector<tileinfer::TokenId> read_prompt(const std::string& input_path)
{
    std::ifstream file(input_path);
    if (!file)
        throw std::runtime_error("cannot open prompt inputs");
    const json request = json::parse(file);
    const auto& tokens = request.at("tokens");
    if (!tokens.is_array() || tokens.empty())
        throw std::runtime_error("prompt requires a nonempty token array");
    std::vector<tileinfer::TokenId> result;
    result.reserve(tokens.size());
    for (const auto& item : tokens)
        result.push_back(parse_token_id(item));
    return result;
}

int option_int(int argc, char** argv, std::string_view key, int start, int fallback)
{
    for (int i = start; i < argc; i += 2) {
        if (i + 1 >= argc)
            throw std::runtime_error("missing option value");
        if (std::string_view(argv[i]) == key)
            return std::stoi(argv[i + 1]);
    }
    return fallback;
}

void validate_options(
    int argc, char** argv, int start, const std::vector<std::string_view>& allowed)
{
    for (int i = start; i < argc; i += 2) {
        if (i + 1 >= argc)
            throw std::runtime_error("missing option value");
        const std::string_view key = argv[i];
        if (std::find(allowed.begin(), allowed.end(), key) == allowed.end())
            throw std::runtime_error("unknown option: " + std::string(key));
    }
}

json generate(const tileinfer::LoadedPlan& plan, const std::string& input_path, int max_new_tokens,
    std::optional<tileinfer::TokenId> eos_id)
{
    const auto prompt = read_prompt(input_path);
    if (max_new_tokens <= 0 || max_new_tokens > plan.max_context - static_cast<int>(prompt.size()))
        throw std::runtime_error("generation exceeds context capacity");
    tileinfer::DecoderModel model(plan);
    auto context = model.create_context();
    auto prefill = context.enqueue_prefill(prompt);
    auto generation = context.enqueue_generate({max_new_tokens, eos_id});
    const auto result = context.wait(generation);
    return {{"prompt_tokens", prompt.size()}, {"generated", result.generated},
        {"generated_count", result.generated.size()}, {"position", result.position},
        {"stop_reason", result.stop_reason == tileinfer::StopReason::eos ? "eos" : "limit"},
        {"compute_us", result.compute_us}, {"plan_sha256", plan.plan_sha256},
        {"weights_sha256", plan.weights_sha256}};
}

json bench(const tileinfer::LoadedPlan& plan, const std::string& input_path, int new_tokens,
    int warmup, int rounds)
{
    if (warmup < 0 || rounds <= 0)
        throw std::runtime_error("invalid benchmark rounds");
    const auto prompt = read_prompt(input_path);
    if (new_tokens <= 0 || new_tokens > plan.max_context - static_cast<int>(prompt.size()))
        throw std::runtime_error("benchmark exceeds context capacity");
    tileinfer::DecoderModel model(plan);
    auto context = model.create_context();
    std::vector<double> samples;
    std::vector<tileinfer::TokenId> expected;
    for (int round = -warmup; round < rounds; ++round) {
        auto reset = context.enqueue_reset();
        auto prefill = context.enqueue_prefill(prompt);
        context.wait(prefill);
        auto generation = context.enqueue_generate({new_tokens, std::nullopt});
        const auto result = context.wait(generation);
        if (expected.empty())
            expected = result.generated;
        else if (expected != result.generated)
            throw std::runtime_error("benchmark generated different tokens across rounds");
        if (round >= 0)
            samples.push_back(result.compute_us);
    }
    const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    return {{"plan_sha256", plan.plan_sha256}, {"weights_sha256", plan.weights_sha256},
        {"weight_dtype",
            plan.weight_dtype == tileinfer::WeightDtype::bfloat16 ? "bfloat16" : "float32"},
        {"prompt_tokens", prompt.size()}, {"generated_count", new_tokens}, {"selection", "greedy"},
        {"warmup", warmup}, {"rounds", rounds}, {"samples_us", samples}, {"mean_us", mean},
        {"us_per_token", mean / new_tokens}, {"tokens_per_second", new_tokens * 1000000.0 / mean},
        {"generated", expected},
        {"timing",
            "CUDA events around GPU selection and decode; setup, prefill, and readback excluded"}};
}

} // namespace

int main(int argc, char** argv)
{
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--help") {
            std::cout << "tile_runtime generate PACKAGE INPUTS.json --max-new-tokens N "
                         "[--eos-id ID]\n"
                         "tile_runtime bench PACKAGE INPUTS.json --new-tokens N "
                         "--warmup W --rounds R\n"
                         "tile_runtime check PACKAGE INPUTS.json\n"
                         "tile_runtime --describe-target-json\n"
                         "tile_runtime --inspect-plan PACKAGE\n";
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
        } else if ((mode == "check" || mode == "--run-stack") &&
                   (argc == 4 || (argc == 5 && std::string_view(argv[4]) == "--trace-blocks"))) {
            std::cout << run_stack(tileinfer::load_plan(argv[2], target), argv[3], argc == 5).dump()
                      << '\n';
#ifdef TILE_TEST_HARNESS
        } else if (mode == "--run-block" && argc == 4) {
            std::cout << run_block(tileinfer::load_plan(argv[2], target), argv[3]).dump() << '\n';
#endif
        } else if (mode == "generate" && argc >= 6) {
            validate_options(argc, argv, 4, {"--max-new-tokens", "--eos-id", "--selection"});
            for (int i = 4; i < argc; i += 2)
                if (std::string_view(argv[i]) == "--selection" &&
                    std::string_view(argv[i + 1]) != "greedy")
                    throw std::runtime_error("only greedy selection is supported");
            const int count = option_int(argc, argv, "--max-new-tokens", 4, -1);
            const int eos = option_int(argc, argv, "--eos-id", 4, -1);
            bool eos_supplied = false;
            for (int i = 4; i < argc; i += 2)
                eos_supplied |= std::string_view(argv[i]) == "--eos-id";
            if (eos_supplied && eos < 0)
                throw std::runtime_error("EOS token ID must be nonnegative");
            std::cout << generate(tileinfer::load_plan(argv[2], target), argv[3], count,
                             eos_supplied ? std::optional<tileinfer::TokenId>(eos) : std::nullopt)
                             .dump()
                      << '\n';
        } else if (mode == "bench" && argc >= 10) {
            validate_options(argc, argv, 4, {"--new-tokens", "--warmup", "--rounds"});
            const int count = option_int(argc, argv, "--new-tokens", 4, -1);
            const int warmup = option_int(argc, argv, "--warmup", 4, -1);
            const int rounds = option_int(argc, argv, "--rounds", 4, -1);
            std::cout << bench(
                             tileinfer::load_plan(argv[2], target), argv[3], count, warmup, rounds)
                             .dump(2)
                      << '\n';
#ifdef TILE_TEST_HARNESS
        } else if (mode == "--benchmark-stack" &&
                   (argc == 8 || (argc == 9 && std::string_view(argv[8]) == "--profile"))) {
            const auto plan = tileinfer::load_plan(argv[2], target);
            const int position = std::stoi(argv[3]);
            const int token_id = std::stoi(argv[4]);
            const int warmup = std::stoi(argv[5]);
            const int iterations = std::stoi(argv[6]);
            const int rounds = std::stoi(argv[7]);
            tileinfer::DiagnosticContext context(plan);
            const auto result =
                context.benchmark_token(position, token_id, warmup, iterations, rounds, argc == 9);
            const double mean =
                std::accumulate(result.samples_us.begin(), result.samples_us.end(), 0.0) /
                result.samples_us.size();
            std::cout << json{{"plan_sha256", plan.plan_sha256},
                             {"weights_sha256", plan.weights_sha256},
                             {"weight_dtype", plan.weight_dtype == tileinfer::WeightDtype::bfloat16
                                                  ? "bfloat16"
                                                  : "float32"},
                             {"position", position}, {"context_length", position + 1},
                             {"token_id", token_id}, {"warmup", warmup}, {"iterations", iterations},
                             {"rounds", rounds}, {"samples_us", result.samples_us},
                             {"mean_us", mean}, {"top_token", result.top_token},
                             {"top_logit", result.top_logit},
                             {"timing", "CUDA event batch average; kernel only"}}
                             .dump(2)
                      << '\n';
        } else if (mode == "--state-smoke" && argc == 3) {
            const auto plan = tileinfer::load_plan(argv[2], target);
            state_smoke(plan);
            std::cout << "{\"state_smoke\": true}\n";
#endif
        } else {
            throw std::runtime_error("unknown mode; use --help");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "tile_runtime: " << error.what() << '\n';
        return 2;
    }
}
