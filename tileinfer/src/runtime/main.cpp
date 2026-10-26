#include "runtime.hpp"
#ifdef TILE_TEST_HARNESS
#include "diagnostic.hpp"
#endif

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
using json = nlohmann::json;

json target_json(const tileinfer::Target& t)
{
    json variants = json::object();
    for (const auto& variant : t.variants)
        variants[variant.name] = {{"registers_per_thread", variant.registers_per_thread},
            {"static_shared_bytes", variant.static_shared_bytes},
            {"active_blocks_per_sm", variant.active_blocks_per_sm},
            {"threads_per_block", tileinfer::kThreads}};
    return {{"device",
                {{"name", t.name}, {"compute_capability", {t.major, t.minor}},
                    {"sm_count", t.sm_count}, {"cooperative_launch", t.cooperative_launch},
                    {"global_memory_bytes", t.global_memory_bytes},
                    {"max_threads_per_sm", t.max_threads_per_sm},
                    {"driver_version", t.driver_version}, {"runtime_version", t.runtime_version}}},
        {"executor", {{"abi", tileinfer::kAbi}, {"opcodes", tileinfer::supported_opcodes()},
                         {"worker_variants", variants}, {"execution_ready", true},
                         {"threads_per_block", tileinfer::kThreads},
                         {"active_blocks_per_sm", t.active_blocks_per_sm},
                         {"registers_per_thread", t.registers_per_thread},
                         {"static_shared_bytes", t.static_shared_bytes},
                         {"attention_shared_bytes", tileinfer::kAttentionSharedBytes},
                         {"fingerprint", tileinfer::kFingerprint}}}};
}

json precision(const tileinfer::LoadedPlan& plan)
{
    const bool bf16 = plan.weight_dtype == tileinfer::WeightDtype::bfloat16;
    const char* dtype = bf16 ? "bfloat16" : "float32";
    return {{"numeric_policy", bf16 ? "hf-llama-eager-bf16-v1" : "decoder-f32-v1"},
        {"weight_dtype", dtype}, {"activation_dtype", dtype}, {"kv_dtype", dtype},
        {"accumulation_dtype", "float32"}, {"logits_compute_dtype", dtype},
        {"api_output_dtype", "float32"}};
}

json inspect(const tileinfer::LoadedPlan& plan)
{
    const auto& shape = plan.shape;
    const std::uint64_t hidden_buffers =
        2ull * shape.hidden_size * tileinfer::element_bytes(plan.weight_dtype);
    const std::uint64_t logits_bytes =
        static_cast<std::uint64_t>(plan.stack.vocab_size) * sizeof(float);
    const std::uint64_t program_bytes = plan.instructions.size() * sizeof(tileinfer::TaskRecord) +
                                        plan.streams.size() * sizeof(tileinfer::WorkerTaskRange) +
                                        plan.rotary_frequencies.size() * sizeof(float);
    const std::uint64_t token_bytes =
        plan.stack.full_stack
            ? 2ull * shape.max_context * sizeof(tileinfer::TokenId) + 2 * sizeof(int)
            : 0;
    const json memory = {{"hidden_buffers_bytes", hidden_buffers},
        {"logits_buffer_bytes", logits_bytes}, {"program_bytes", program_bytes},
        {"token_status_bytes", token_bytes},
        {"context_device_payload_bytes",
            hidden_buffers + logits_bytes + token_bytes + plan.scratch_bytes + plan.kv_bytes +
                plan.attention_bytes + plan.attention_partial_bytes + plan.ffn_partial_bytes},
        {"model_device_payload_bytes", plan.weights.size() + program_bytes},
        {"allocator_runtime_overhead_bytes", nullptr},
        {"scope",
            "validated payload extents including alignment; allocator/runtime overhead and "
            "optional diagnostic snapshots excluded"}};
    return {{"plan_sha256", plan.plan_sha256}, {"weights_sha256", plan.weights_sha256},
        {"memory", memory},
        {"model",
            {{"hidden_size", shape.hidden_size}, {"intermediate_size", shape.intermediate_size},
                {"num_attention_heads", shape.query_heads}, {"num_key_value_heads", shape.kv_heads},
                {"head_dim", shape.head_dim}, {"max_context", shape.max_context},
                {"attention_chunks", shape.attention_chunks},
                {"attention_chunk_steps", shape.attention_chunk_steps},
                {"attention_fused_steps", shape.attention_fused_steps},
                {"ffn_chunks", shape.ffn_chunks}}},
        {"workers", plan.workers}, {"wave_count", plan.wave_count}, {"task_count", plan.task_count},
        {"instruction_count", plan.instructions.size()}, {"worker_loads", plan.worker_loads},
        {"scratch_bytes", plan.scratch_bytes}, {"attention_bytes", plan.attention_bytes},
        {"attention_partial_bytes", plan.attention_partial_bytes},
        {"ffn_partial_bytes", plan.ffn_partial_bytes}, {"kv_bytes", plan.kv_bytes},
        {"weight_bytes", plan.weights.size()},
        {"weight_dtype",
            plan.weight_dtype == tileinfer::WeightDtype::bfloat16 ? "bfloat16" : "float32"},
        {"model_kind", plan.stack.full_stack ? "llama_decoder_stack" : "llama_decoder_block"},
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
        auto segment = [&](tileinfer::TensorId tensor, int count) {
            const auto offset = plan.tensors.at(tileinfer::tensor_index(tensor)).offset;
            return std::vector<float>(scratch.begin() + offset, scratch.begin() + offset + count);
        };
        using tileinfer::TensorId;
        const int kv_width = plan.shape.kv_heads * plan.shape.head_dim;
        const int p = context.position() - 1;
        const auto key_offset = plan.layout.key + p * kv_width;
        const auto value_offset = plan.layout.value + p * kv_width;
        const std::vector<float> key(kv.begin() + key_offset, kv.begin() + key_offset + kv_width);
        const std::vector<float> value(
            kv.begin() + value_offset, kv.begin() + value_offset + kv_width);
        auto residual = segment(TensorId::r, plan.shape.hidden_size);
        const auto ffn = context.ffn_snapshot();
        steps.push_back(
            {{"position", p}, {"q", segment(TensorId::q_rot, plan.shape.hidden_size)}, {"k", key},
                {"v", value}, {"attention", segment(TensorId::attention, plan.shape.hidden_size)},
                {"attn_residual", residual}, {"ffn", ffn}, {"y", output}});
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
    const int width = plan.shape.kv_heads * plan.shape.head_dim;
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
            const int key_offset = base + plan.layout.key;
            const int value_offset = base + plan.layout.value;
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
                const auto begin = hidden.begin() + layer * plan.shape.hidden_size;
                blocks.push_back(std::vector<float>(begin, begin + plan.shape.hidden_size));
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
    for (int position = 0; position < plan.shape.max_context; ++position) {
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

int parse_int(std::string_view text)
{
    int result = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), result);
    if (error != std::errc{} || end != text.data() + text.size())
        throw std::runtime_error("invalid integer: " + std::string(text));
    return result;
}

enum class Mode
{
    help,
    target,
    inspect,
    check,
    generate,
    bench,
    block,
    state,
    fixed_bench,
    serve,
    timeline
};

struct Command
{
    Mode mode;
    std::string package;
    std::string input;
    int count = 0;
    int warmup = 0;
    int rounds = 0;
    int position = 0;
    int token = 0;
    int iterations = 0;
    std::vector<int> positions; // timeline: in the order traced
    bool synthetic = false;     // fixed-position modes: synthetic K/V prefix
    std::optional<tileinfer::TokenId> eos;
    bool trace = false;
    bool profile = false;
    Command(Mode value, std::string package_path = {}, std::string input_path = {})
        : mode(value)
        , package(std::move(package_path))
        , input(std::move(input_path))
    {
    }
};

Command parse_command(int argc, char** argv)
{
    if (argc < 2)
        throw std::runtime_error("missing mode; use --help");
    const std::string_view mode = argv[1];
    if (argc == 2 && mode == "--help")
        return {Mode::help};
    if (argc == 2 && mode == "--describe-target-json")
        return {Mode::target};
    if (argc == 3 && mode == "--inspect-plan")
        return {Mode::inspect, argv[2]};
    if ((mode == "check" || mode == "--run-stack") &&
        (argc == 4 || (argc == 5 && std::string_view(argv[4]) == "--trace-blocks"))) {
        Command result{Mode::check, argv[2], argv[3]};
        result.trace = argc == 5;
        return result;
    }
#ifdef TILE_TEST_HARNESS
    if (mode == "--run-block" && argc == 4)
        return {Mode::block, argv[2], argv[3]};
    if (mode == "--state-smoke" && argc == 3)
        return {Mode::state, argv[2]};
    if (mode == "--benchmark-stack" &&
        (argc == 8 || (argc == 9 && std::string_view(argv[8]) == "--profile"))) {
        Command result{Mode::fixed_bench, argv[2]};
        result.position = parse_int(argv[3]);
        result.token = parse_int(argv[4]);
        result.warmup = parse_int(argv[5]);
        result.iterations = parse_int(argv[6]);
        result.rounds = parse_int(argv[7]);
        result.profile = argc == 9;
        if (result.position < 0 || result.token < 0 || result.warmup < 0 ||
            result.iterations <= 0 || result.rounds <= 0)
            throw std::runtime_error("invalid benchmark arguments");
        return result;
    }
    const bool synthetic = std::string_view(argv[argc - 1]) == "--synthetic-prefix";
    if (mode == "--benchmark-serve" && argc - synthetic == 3) {
        Command result{Mode::serve, argv[2]};
        result.synthetic = synthetic;
        return result;
    }
    if (mode == "--timeline-stack" && argc - synthetic == 7) {
        Command result{Mode::timeline, argv[2]};
        for (std::string_view list = argv[3]; !list.empty();) {
            const auto comma = list.find(',');
            result.positions.push_back(parse_int(list.substr(0, comma)));
            list = comma == std::string_view::npos ? std::string_view{} : list.substr(comma + 1);
        }
        result.token = parse_int(argv[4]);
        result.warmup = parse_int(argv[5]);
        result.rounds = parse_int(argv[6]);
        result.synthetic = synthetic;
        if (result.positions.empty() ||
            std::ranges::any_of(result.positions, [](int p) { return p < 0; }) ||
            result.token < 0 || result.warmup < 0 || result.rounds <= 0)
            throw std::runtime_error("invalid timeline arguments");
        return result;
    }
#endif
    if ((mode != "generate" && mode != "bench") || argc < 4)
        throw std::runtime_error("unknown mode or invalid arguments; use --help");
    const bool generation = mode == "generate";
    Command result{generation ? Mode::generate : Mode::bench, argv[2], argv[3]};
    std::set<std::string_view> seen;
    for (int i = 4; i < argc; i += 2) {
        if (i + 1 >= argc)
            throw std::runtime_error("missing option value");
        const std::string_view key = argv[i];
        if (!seen.insert(key).second)
            throw std::runtime_error("duplicate option: " + std::string(key));
        if (key == (generation ? "--max-new-tokens" : "--new-tokens"))
            result.count = parse_int(argv[i + 1]);
        else if (generation && key == "--eos-id") {
            result.eos = parse_int(argv[i + 1]);
            if (*result.eos < 0)
                throw std::runtime_error("EOS token ID must be nonnegative");
        } else if (generation && key == "--selection") {
            if (std::string_view(argv[i + 1]) != "greedy")
                throw std::runtime_error("only greedy selection is supported");
        } else if (!generation && key == "--warmup")
            result.warmup = parse_int(argv[i + 1]);
        else if (!generation && key == "--rounds")
            result.rounds = parse_int(argv[i + 1]);
        else
            throw std::runtime_error("unknown option: " + std::string(key));
    }
    if (!seen.contains(generation ? "--max-new-tokens" : "--new-tokens") ||
        (!generation && (!seen.contains("--warmup") || !seen.contains("--rounds"))))
        throw std::runtime_error("missing required option");
    if (result.count <= 0 || (!generation && (result.warmup < 0 || result.rounds <= 0)))
        throw std::runtime_error("invalid generation or benchmark arguments");
    return result;
}

json generate(const tileinfer::LoadedPlan& plan, const std::string& input_path, int max_new_tokens,
    std::optional<tileinfer::TokenId> eos_id)
{
    const auto prompt = read_prompt(input_path);
    if (prompt.size() > static_cast<std::size_t>(plan.shape.max_context) || max_new_tokens <= 0 ||
        max_new_tokens > plan.shape.max_context - static_cast<int>(prompt.size()))
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
    if (prompt.size() > static_cast<std::size_t>(plan.shape.max_context) || new_tokens <= 0 ||
        new_tokens > plan.shape.max_context - static_cast<int>(prompt.size()))
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

#ifdef TILE_TEST_HARNESS
json fixed_benchmark(const tileinfer::LoadedPlan& plan, tileinfer::DiagnosticContext& context,
    int position, int token, int warmup, int iterations, int rounds, bool profile)
{
    const auto measurement =
        context.benchmark_token(position, token, warmup, iterations, rounds, profile);
    const double mean =
        std::accumulate(measurement.samples_us.begin(), measurement.samples_us.end(), 0.0) /
        measurement.samples_us.size();
    json result = {{"plan_sha256", plan.plan_sha256}, {"weights_sha256", plan.weights_sha256},
        {"weights_verified", plan.weights_verified}, {"position", position},
        {"context_length", position + 1}, {"token_id", token}, {"warmup", warmup},
        {"iterations", iterations}, {"rounds", rounds}, {"samples_us", measurement.samples_us},
        {"mean_us", mean}, {"top_token", measurement.top_token},
        {"top_logit", measurement.top_logit}, {"timing", "CUDA event batch average; kernel only"}};
    result.update(precision(plan));
    return result;
}

// Answers fixed-position benchmark requests, one per input line
// "POSITION TOKEN WARMUP ITERATIONS ROUNDS", with one JSON line each, until
// end of input. The package stays resident and the context keeps its filled
// prefix, so interleaved measurements of several packages need no reloads.
// A synthetic prefix skips the prefix decode; its logits are not a token's.
void serve_benchmarks(const tileinfer::LoadedPlan& plan, bool synthetic)
{
    tileinfer::DiagnosticContext context(plan);
    if (synthetic)
        context.use_synthetic_prefix();
    std::cout << json{{"ready", true}, {"plan_sha256", plan.plan_sha256},
                     {"weights_verified", plan.weights_verified}}
                     .dump()
              << std::endl;
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream request(line);
        int position = 0, token = 0, warmup = 0, iterations = 0, rounds = 0;
        if (!(request >> position >> token >> warmup >> iterations >> rounds) || iterations <= 0 ||
            rounds <= 0)
            throw std::runtime_error("invalid benchmark request: " + line);
        auto result =
            fixed_benchmark(plan, context, position, token, warmup, iterations, rounds, false);
        result["synthetic_prefix"] = synthetic;
        std::cout << result.dump() << std::endl;
    }
}
#endif

int main(int argc, char** argv)
{
    try {
        const Command command = parse_command(argc, argv);
        if (command.mode == Mode::help) {
            std::cout
                << "tile_runtime generate PACKAGE INPUTS.json --max-new-tokens N [--eos-id ID]\n"
                   "tile_runtime bench PACKAGE INPUTS.json --new-tokens N --warmup W --rounds R\n"
                   "tile_runtime check PACKAGE INPUTS.json [--trace-blocks]\n"
                   "tile_runtime --describe-target-json\n"
                   "tile_runtime --inspect-plan PACKAGE\n";
            return 0;
        }
        const auto target = tileinfer::describe_target();
        if (command.mode == Mode::target) {
            std::cout << target_json(target).dump(2) << '\n';
            return 0;
        }
        // Measurement modes reload a package many times per run; they trust its
        // declared weight checksum instead of hashing 2.5 GB each time.
        const bool measurement = command.mode == Mode::fixed_bench || command.mode == Mode::serve ||
                                 command.mode == Mode::timeline;
        const auto plan = tileinfer::load_plan(command.package, target, !measurement);
        json result;
        switch (command.mode) {
        case Mode::inspect:
            result = inspect(plan);
            break;
        case Mode::check:
            result = run_stack(plan, command.input, command.trace);
            break;
        case Mode::generate:
            result = generate(plan, command.input, command.count, command.eos);
            break;
        case Mode::bench:
            result = bench(plan, command.input, command.count, command.warmup, command.rounds);
            break;
#ifdef TILE_TEST_HARNESS
        case Mode::block:
            result = run_block(plan, command.input);
            break;
        case Mode::state:
            state_smoke(plan);
            result = {{"state_smoke", true}};
            break;
        case Mode::fixed_bench: {
            tileinfer::DiagnosticContext context(plan);
            result = fixed_benchmark(plan, context, command.position, command.token, command.warmup,
                command.iterations, command.rounds, command.profile);
            break;
        }
        case Mode::serve:
            serve_benchmarks(plan, command.synthetic);
            return 0;
        case Mode::timeline: {
            // One JSON line per position; the context keeps its filled prefix.
            tileinfer::DiagnosticContext context(plan);
            if (command.synthetic)
                context.use_synthetic_prefix();
            std::vector<int> workers;
            for (std::size_t worker = 0; worker < plan.streams.size(); ++worker)
                workers.insert(workers.end(), plan.streams[worker].count, static_cast<int>(worker));
            for (const int position : command.positions) {
                const auto timeline =
                    context.timeline_token(position, command.token, command.warmup, command.rounds);
                json samples = json::array();
                for (std::size_t i = 0; i < timeline.records.size(); ++i) {
                    std::array<std::vector<std::uint64_t>, 3> fields;
                    for (std::size_t word = 0; word < timeline.records[i].size(); ++word)
                        fields[word % 3].push_back(timeline.records[i][word]);
                    samples.push_back(
                        {{"kernel_us", timeline.kernel_us[i]}, {"start_cycle", fields[0]},
                            {"end_cycle", fields[1]}, {"start_ns", fields[2]}});
                }
                json trace = {{"plan_sha256", plan.plan_sha256},
                    {"weights_sha256", plan.weights_sha256},
                    {"weights_verified", plan.weights_verified}, {"position", position},
                    {"context_length", position + 1}, {"token_id", command.token},
                    {"warmup", command.warmup}, {"synthetic_prefix", command.synthetic},
                    {"instruction_tasks", plan.instruction_tasks}, {"instruction_workers", workers},
                    {"samples", samples},
                    {"timing",
                        "SM cycle counters per instruction; %globaltimer at instruction start for "
                        "cross-worker alignment; kernel_us is the CUDA event time of the traced "
                        "launch"}};
                trace.update(precision(plan));
                std::cout << trace.dump() << std::endl;
            }
            return 0;
        }
#endif
        default:
            throw std::runtime_error("unsupported command");
        }
        if (command.mode != Mode::state)
            result.update(precision(plan));
        std::cout << result.dump() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "tile_runtime: " << error.what() << '\n';
        return 2;
    }
}
