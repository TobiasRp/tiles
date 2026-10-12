#include "runtime.hpp"

#include <picosha2.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string_view>

namespace tileinfer {
namespace {
using json = nlohmann::json;

void require(bool condition, std::string_view message)
{
    if (!condition)
        throw std::runtime_error(std::string(message));
}

std::vector<std::uint8_t> read_bytes(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    require(static_cast<bool>(file), "cannot open " + path.string());
    auto size = file.tellg();
    require(size >= 0, "cannot size " + path.string());
    std::vector<std::uint8_t> result(static_cast<size_t>(size));
    file.seekg(0);
    if (!result.empty())
        file.read(reinterpret_cast<char*>(result.data()), size);
    require(static_cast<bool>(file), "cannot read " + path.string());
    return result;
}

int positive(const json& value, const char* name)
{
    require(value.is_number_integer(), std::string("invalid ") + name);
    auto n = value.get<std::int64_t>();
    require(n > 0 && n <= std::numeric_limits<int>::max(), std::string("invalid ") + name);
    return static_cast<int>(n);
}

int nonnegative(const json& value, const char* name)
{
    require(value.is_number_integer(), std::string("invalid ") + name);
    auto n = value.get<std::int64_t>();
    require(n >= 0 && n <= std::numeric_limits<int>::max(), std::string("invalid ") + name);
    return static_cast<int>(n);
}

json tensor(int id, const char* name, json shape, json strides, const char* storage)
{
    return {{"id", id}, {"name", name}, {"shape", std::move(shape)},
        {"strides", std::move(strides)}, {"dtype", "float32"}, {"storage", storage}};
}

struct Stage
{
    const char* opcode;
    const char* input;
    const char* weight;
    const char* bias;
    const char* output;
    int rows;
    int width;
    int tile;
    Opcode code;
    Buffer input_buffer;
    Buffer output_buffer;
    int weight_offset;
    int bias_offset;
};

} // namespace

std::string sha256(const std::vector<std::uint8_t>& bytes)
{
    return picosha2::hash256_hex_string(bytes);
}

Plan load_plan(const std::string& package, const Target& device)
{
    const auto plan_data = read_bytes(std::filesystem::path(package) / "plan.json");
    const auto weights = read_bytes(std::filesystem::path(package) / "weights.bin");
    const auto document = json::parse(plan_data.begin(), plan_data.end());
    const auto& header = document.at("header");
    require(header.at("format_version") == 1 && header.at("executor_abi") == kAbi &&
                header.at("model_kind") == "mlp_f32" && header.at("activation") == "gelu_tanh",
        "unsupported plan header");

    const auto& model = document.at("model");
    const int d = positive(model.at("D"), "D");
    const int h = positive(model.at("H"), "H");
    const int o = positive(model.at("O"), "O");
    const int t1 = positive(model.at("tile_h1"), "tile_h1");
    const int t2 = positive(model.at("tile_h2"), "tile_h2");
    require(model.at("dtype") == "float32", "unsupported model dtype");
    const auto& target = document.at("target");
    const int workers = positive(target.at("worker_count"), "worker_count");
    require(target.at("compute_capability") == json::array({device.major, device.minor}) &&
                target.at("sm_count") == device.sm_count && workers == device.sm_count &&
                target.at("cooperative_launch") == true && device.cooperative_launch &&
                target.at("threads_per_block") == kThreads &&
                target.at("kernel_fingerprint") == kFingerprint &&
                workers <= device.sm_count * device.active_blocks_per_sm,
        "plan target does not match the compiled worker or cooperative device");
    require(static_cast<std::int64_t>(d) * h <= std::numeric_limits<int>::max() &&
                static_cast<std::int64_t>(h) * o <= std::numeric_limits<int>::max(),
        "model too large for task offsets");

    const json tensors = json::array({tensor(0, "x", {d}, {1}, "input"),
        tensor(1, "W1", {h, d}, {d, 1}, "weight"), tensor(2, "b1", {h}, {1}, "weight"),
        tensor(3, "h", {h}, {1}, "scratch"), tensor(4, "W2", {o, h}, {h, 1}, "weight"),
        tensor(5, "b2", {o}, {1}, "weight"), tensor(6, "y", {o}, {1}, "output")});
    require(model.at("tensors") == tensors, "invalid tensor descriptors");

    const auto& storage = document.at("storage");
    require(storage.at("input_bytes") == static_cast<std::int64_t>(d) * 4 &&
                storage.at("output_bytes") == static_cast<std::int64_t>(o) * 4 &&
                storage.at("scratch_bytes") == static_cast<std::int64_t>(h) * 4 &&
                storage.at("scratch") ==
                    json::array({{{"tensor", "h"}, {"offset", 0},
                        {"bytes", static_cast<std::int64_t>(h) * 4}, {"alignment", 4}}}),
        "invalid runtime storage");
    const auto& integrity = document.at("integrity");
    require(storage.at("weight_bytes") == weights.size() &&
                integrity.at("weights_bytes") == weights.size() &&
                integrity.at("weights_sha256") == sha256(weights),
        "weight length or checksum mismatch");
    const auto& entries = storage.at("weights");
    require(entries.is_array() && entries.size() == 4, "expected four weight regions");
    const char* names[] = {"W1", "b1", "W2", "b2"};
    const std::int64_t sizes[] = {static_cast<std::int64_t>(h) * d * 4,
        static_cast<std::int64_t>(h) * 4, static_cast<std::int64_t>(o) * h * 4,
        static_cast<std::int64_t>(o) * 4};
    std::int64_t cursor = 0;
    int offsets[4]{};
    for (int i = 0; i < 4; ++i) {
        const int offset = nonnegative(entries[i].at("offset"), "weight offset");
        require(entries[i] == json{{"tensor", names[i]}, {"offset", offset}, {"bytes", sizes[i]},
                                  {"alignment", 64}} &&
                    offset >= cursor && offset % 64 == 0 &&
                    sizes[i] <= static_cast<std::int64_t>(weights.size()) - offset,
            "invalid weight region");
        offsets[i] = offset / 4;
        cursor = offset + sizes[i];
    }
    require(cursor == static_cast<std::int64_t>(weights.size()), "trailing weight data");

    const Stage stages[] = {
        {"gemv_bias_gelu_f32", "x", "W1", "b1", "h", h, d, t1, Opcode::gemv_bias_gelu_f32,
            Buffer::input, Buffer::scratch, offsets[0], offsets[1]},
        {"gemv_bias_f32", "h", "W2", "b2", "y", o, h, t2, Opcode::gemv_bias_f32, Buffer::scratch,
            Buffer::output, offsets[2], offsets[3]},
    };
    const auto& tasks = document.at("tasks");
    const auto& waves = document.at("waves");
    require(tasks.is_array() && waves.is_array() && waves.size() == 2,
        "MLP requires two scheduled waves");
    const int first_count = (h - 1) / t1 + 1;
    const int second_count = (o - 1) / t2 + 1;
    require(
        tasks.size() == static_cast<size_t>(first_count + second_count), "incorrect task count");
    std::vector<Task> encoded(tasks.size());
    for (int stage_index = 0, id = 0; stage_index < 2; ++stage_index) {
        const Stage& stage = stages[stage_index];
        const int count = stage_index == 0 ? first_count : second_count;
        for (int tile = 0; tile < count; ++tile, ++id) {
            const auto& task = tasks[id];
            const int begin = tile * stage.tile;
            const int end = std::min(begin + stage.tile, stage.rows);
            const int worker = nonnegative(task.at("worker"), "worker");
            require(worker < workers && task.at("id") == id && task.at("wave") == stage_index &&
                        task.at("opcode") == stage.opcode &&
                        task.at("rows") == json::array({begin, end}) &&
                        task.at("input") == stage.input && task.at("weight") == stage.weight &&
                        task.at("bias") == stage.bias && task.at("output") == stage.output,
                "invalid task descriptor");
            json expected_predecessors = json::array();
            if (stage_index == 1)
                for (int producer = 0; producer < first_count; ++producer)
                    expected_predecessors.push_back(producer);
            require(task.at("predecessors") == expected_predecessors, "invalid task dependencies");
            encoded[id] = {static_cast<unsigned>(stage.code), static_cast<unsigned>(begin),
                static_cast<unsigned>(end), static_cast<unsigned>(stage.width),
                static_cast<unsigned>(stage.weight_offset),
                static_cast<unsigned>(stage.bias_offset), static_cast<unsigned>(stage.input_buffer),
                static_cast<unsigned>(stage.output_buffer)};
        }
    }

    Plan result{d, h, o, workers, 2, weights, {}, {}, sha256(plan_data), sha256(weights)};
    std::vector<int> seen(tasks.size());
    for (int wave_index = 0; wave_index < 2; ++wave_index) {
        const auto& wave = waves[wave_index];
        require(wave.at("id") == wave_index && wave.at("workers").is_array() &&
                    wave.at("workers").size() == static_cast<size_t>(workers) &&
                    wave.at("estimated_work").is_array() &&
                    wave.at("estimated_work").size() == static_cast<size_t>(workers),
            "incomplete worker schedule");
        for (int worker = 0; worker < workers; ++worker) {
            const auto& ids = wave.at("workers")[worker];
            require(ids.is_array(), "invalid worker task list");
            const auto begin = result.tasks.size();
            std::int64_t work = 0;
            for (const auto& raw_id : ids) {
                const int id = nonnegative(raw_id, "scheduled task ID");
                require(id < static_cast<int>(tasks.size()) && tasks[id].at("wave") == wave_index &&
                            tasks[id].at("worker") == worker && seen[id]++ == 0,
                    "task assigned incorrectly or twice");
                result.tasks.push_back(encoded[id]);
                work += static_cast<std::int64_t>(encoded[id].row_end - encoded[id].row_begin) *
                        encoded[id].width;
            }
            require(wave.at("estimated_work")[worker] == work, "incorrect worker work estimate");
            result.ranges.push_back(
                {static_cast<unsigned>(begin), static_cast<unsigned>(result.tasks.size() - begin)});
        }
    }
    require(std::all_of(seen.begin(), seen.end(), [](int count) { return count == 1; }),
        "missing scheduled task");
    return result;
}

} // namespace tileinfer
