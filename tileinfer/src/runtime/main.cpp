#include "runtime.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
using json = nlohmann::json;

void require(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}

std::string option(int argc, char** argv, std::string_view name, bool required = true) {
    for (int i = 2; i + 1 < argc; ++i)
        if (std::string_view(argv[i]) == name) return argv[i + 1];
    if (required) throw std::runtime_error("missing option " + std::string(name));
    return {};
}

int count_option(int argc, char** argv, std::string_view name, int default_value) {
    const auto text = option(argc, argv, name, false);
    if (text.empty()) return default_value;
    size_t consumed = 0;
    const int value = std::stoi(text, &consumed);
    require(consumed == text.size() && value > 0, "invalid positive count");
    return value;
}

std::vector<std::uint8_t> read_bytes(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    require(static_cast<bool>(file), "cannot open input file");
    const auto size = file.tellg();
    require(size >= 0, "cannot size input file");
    std::vector<std::uint8_t> result(static_cast<size_t>(size));
    file.seekg(0);
    if (!result.empty()) file.read(reinterpret_cast<char*>(result.data()), size);
    require(static_cast<bool>(file), "cannot read input file");
    return result;
}

void write_bytes(const std::string& path, const std::vector<float>& values) {
    std::ofstream file(path, std::ios::binary);
    require(static_cast<bool>(file), "cannot open output file");
    file.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
    require(static_cast<bool>(file), "cannot write output file");
}

void write_json(const std::string& path, const json& value) {
    std::ofstream file(path);
    require(static_cast<bool>(file), "cannot open report file");
    file << value.dump(2) << '\n';
    require(static_cast<bool>(file), "cannot write report file");
}

json target_json(const tileinfer::Target& t) {
    return {
        {"device", {{"name", t.name}, {"compute_capability", {t.major, t.minor}},
                    {"sm_count", t.sm_count}, {"cooperative_launch", t.cooperative_launch},
                    {"global_memory_bytes", t.global_memory_bytes}, {"max_threads_per_sm", t.max_threads_per_sm},
                    {"driver_version", t.driver_version}, {"runtime_version", t.runtime_version}}},
        {"executor", {{"abi", tileinfer::kAbi},
                      {"opcodes", {"gemv_bias_gelu_f32", "gemv_bias_f32"}},
                      {"threads_per_block", tileinfer::kThreads},
                      {"active_blocks_per_sm", t.active_blocks_per_sm},
                      {"registers_per_thread", t.registers_per_thread},
                      {"static_shared_bytes", t.static_shared_bytes},
                      {"fingerprint", tileinfer::kFingerprint}}}
    };
}

json identity(const tileinfer::Plan& p) {
    return {{"plan_sha256", p.plan_sha256}, {"weights_sha256", p.weights_sha256},
            {"dimensions", {{"D", p.d}, {"H", p.h}, {"O", p.o}}},
            {"workers", p.workers}, {"waves", p.wave_count}, {"tasks", p.tasks.size()}};
}

void run_cases(int argc, char** argv, const tileinfer::Target& target) {
    const auto plan = tileinfer::load_plan(argv[2], target);
    const auto input_data = read_bytes(option(argc, argv, "--inputs"));
    require(!input_data.empty() && input_data.size() % (static_cast<size_t>(plan.d) * 4) == 0,
            "input length must contain complete float32 vectors");
    const size_t cases = input_data.size() / (static_cast<size_t>(plan.d) * 4);
    std::vector<float> x(static_cast<size_t>(plan.d) * cases);
    std::memcpy(x.data(), input_data.data(), input_data.size());
    std::vector<float> y(static_cast<size_t>(plan.o) * cases);
    const auto hidden_path = option(argc, argv, "--hidden", false);
    std::vector<float> h(hidden_path.empty() ? 0 : static_cast<size_t>(plan.h) * cases);
    tileinfer::Executor executor(plan);
    const auto began = std::chrono::steady_clock::now();
    for (size_t i = 0; i < cases; ++i)
        executor.run(x.data() + i * plan.d, y.data() + i * plan.o,
                     h.empty() ? nullptr : h.data() + i * plan.h);
    const double elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - began).count();
    write_bytes(option(argc, argv, "--outputs"), y);
    if (!hidden_path.empty()) write_bytes(hidden_path, h);
    auto report = identity(plan);
    report["case_count"] = cases;
    report["input_sha256"] = tileinfer::sha256(input_data);
    report["host_to_host_ms_total"] = elapsed_ms;
    report["pass"] = true;
    write_json(option(argc, argv, "--report"), report);
}

void benchmark(int argc, char** argv, const tileinfer::Target& target) {
    const auto plan = tileinfer::load_plan(argv[2], target);
    const auto input_data = read_bytes(option(argc, argv, "--input"));
    require(input_data.size() == static_cast<size_t>(plan.d) * 4, "benchmark input length mismatch");
    std::vector<float> x(plan.d);
    std::memcpy(x.data(), input_data.data(), input_data.size());
    const int iterations = count_option(argc, argv, "--iterations", 100);
    const int rounds = count_option(argc, argv, "--rounds", 5);
    tileinfer::Executor executor(plan);
    executor.upload_input(x.data());
    for (int i = 0; i < 20; ++i) executor.launch();
    executor.synchronize();
    std::vector<double> samples;
    for (int i = 0; i < rounds; ++i) samples.push_back(executor.timed_batch(iterations));
    const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    auto report = identity(plan);
    report["input_sha256"] = tileinfer::sha256(input_data);
    report["iterations_per_round"] = iterations;
    report["rounds"] = rounds;
    report["gpu_batch_mean_us_per_forward"] = mean;
    report["gpu_batch_rounds_us_per_forward"] = samples;
    report["device"] = target.name;
    write_json(option(argc, argv, "--report"), report);
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--help") {
            std::cout << "tile_runtime --describe-target-json\n"
                         "tile_runtime --run-plan PACKAGE --inputs FILE --outputs FILE [--hidden FILE] --report FILE\n"
                         "tile_runtime --benchmark PACKAGE --input FILE --iterations N --rounds R --report FILE\n";
            return 0;
        }
        require(argc >= 2, "missing mode; use --help");
        const std::string_view mode = argv[1];
        const auto target = tileinfer::describe_target();
        if (mode == "--describe-target-json") {
            std::cout << target_json(target).dump(2) << '\n';
        } else if (mode == "--run-plan" && argc >= 3) {
            run_cases(argc, argv, target);
        } else if (mode == "--benchmark" && argc >= 3) {
            benchmark(argc, argv, target);
        } else {
            throw std::runtime_error("unknown mode; use --help");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "tile_runtime: " << error.what() << '\n';
        return 2;
    }
}
