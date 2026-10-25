// Probe for the runtime comparison: greedy decode on token ids and fixed-
// context decode timing, with and without sampling. JSON goes to stdout,
// llama.cpp logs go to stderr.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "llama.h"

namespace {

using clock_type = std::chrono::steady_clock;

struct Options {
    std::string mode;
    std::string model;
    std::string kv = "bf16";
    std::string flash_attn = "auto";
    int gpu_layers = 99;
    int threads = 0;
    std::vector<llama_token> ids;
    std::vector<llama_token> forced;  // teacher forcing: tokens to feed instead of the sampled ones
    std::vector<int> positions{0, 63, 255};
    int new_tokens = 32;
    llama_token token = 1;
    int warmup = 3;
    int iterations = 5;
    int rounds = 3;
};

std::vector<long> parse_list(const char * text) {
    std::vector<long> values;
    for (const char * p = text; *p;) {
        char * end = nullptr;
        values.push_back(std::strtol(p, &end, 10));
        p = *end == ',' ? end + 1 : end;
        if (end == p && *p) {
            throw std::runtime_error("bad list");
        }
    }
    return values;
}

ggml_type parse_kv(const std::string & name) {
    if (name == "bf16") return GGML_TYPE_BF16;
    if (name == "f16") return GGML_TYPE_F16;
    if (name == "f32") return GGML_TYPE_F32;
    throw std::runtime_error("unsupported KV type " + name);
}

llama_flash_attn_type parse_fa(const std::string & name) {
    if (name == "on") return LLAMA_FLASH_ATTN_TYPE_ENABLED;
    if (name == "off") return LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if (name == "auto") return LLAMA_FLASH_ATTN_TYPE_AUTO;
    throw std::runtime_error("flash attention must be on, off or auto");
}

Options parse(int argc, char ** argv) {
    Options o;
    if (argc < 2) {
        throw std::runtime_error("usage: rt_probe greedy|decode --model M [options]");
    }
    o.mode = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
            return argv[++i];
        };
        if (a == "--model") o.model = next();
        else if (a == "--kv") o.kv = next();
        else if (a == "--fa") o.flash_attn = next();
        else if (a == "--ngl") o.gpu_layers = std::atoi(next());
        else if (a == "--threads") o.threads = std::atoi(next());
        else if (a == "--ids") for (long v : parse_list(next())) o.ids.push_back((llama_token) v);
        else if (a == "--force") for (long v : parse_list(next())) o.forced.push_back((llama_token) v);
        else if (a == "--positions") {
            o.positions.clear();
            for (long v : parse_list(next())) o.positions.push_back((int) v);
        }
        else if (a == "--new-tokens") o.new_tokens = std::atoi(next());
        else if (a == "--token") o.token = std::atoi(next());
        else if (a == "--warmup") o.warmup = std::atoi(next());
        else if (a == "--iterations") o.iterations = std::atoi(next());
        else if (a == "--rounds") o.rounds = std::atoi(next());
        else throw std::runtime_error("unknown option " + a);
    }
    if (o.model.empty()) throw std::runtime_error("--model is required");
    return o;
}

double elapsed_us(clock_type::time_point start) {
    return std::chrono::duration<double, std::micro>(clock_type::now() - start).count();
}

struct Top {
    llama_token id;
    float logit;
};

Top argmax(const float * logits, int n) {
    const float * best = std::max_element(logits, logits + n);
    return {(llama_token) (best - logits), *best};
}

void check(int status, const char * what) {
    if (status != 0) throw std::runtime_error(std::string(what) + " failed: " + std::to_string(status));
}

// One-token decode at the next free position; the logits are ready on return.
void decode_one(llama_context * ctx, llama_token token) {
    llama_batch batch = llama_batch_get_one(&token, 1);
    check(llama_decode(ctx, batch), "llama_decode");
    llama_synchronize(ctx);
}

void print_array(const char * name, const std::vector<double> & values) {
    std::printf("\"%s\":[", name);
    for (size_t i = 0; i < values.size(); ++i) {
        std::printf("%s%.3f", i ? "," : "", values[i]);
    }
    std::printf("]");
}

struct Session {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    const llama_vocab * vocab = nullptr;
    int n_vocab = 0;

    Session(const Options & o, int n_ctx) {
        llama_log_set([](ggml_log_level level, const char * text, void *) {
            if (level >= GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_NONE) std::fputs(text, stderr);
        }, nullptr);
        llama_backend_init();
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = o.gpu_layers;
        model = llama_model_load_from_file(o.model.c_str(), mp);
        if (!model) throw std::runtime_error("cannot load " + o.model);
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = n_ctx;
        cp.n_batch = 512;
        cp.n_ubatch = 512;
        cp.type_k = parse_kv(o.kv);
        cp.type_v = parse_kv(o.kv);
        cp.flash_attn_type = parse_fa(o.flash_attn);
        if (o.threads > 0) {
            cp.n_threads = o.threads;
            cp.n_threads_batch = o.threads;
        }
        cp.no_perf = true;
        ctx = llama_init_from_model(model, cp);
        if (!ctx) throw std::runtime_error("cannot create context");
        vocab = llama_model_get_vocab(model);
        n_vocab = llama_vocab_n_tokens(vocab);
    }

    ~Session() {
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
        llama_backend_free();
    }

    Top top() const { return argmax(llama_get_logits_ith(ctx, -1), n_vocab); }
};

// One generation: prompt fed token by token, then greedy selection + decode.
struct Generation {
    std::vector<llama_token> tokens;
    std::vector<double> top_logits;
    double decode_us_per_token = 0;
    double sample_us_per_token = 0;
};

Generation generate(Session & s, const Options & o, llama_sampler * greedy) {
    llama_memory_clear(llama_get_memory(s.ctx), true);
    // The prompt is fed one token at a time, like TileInfer and the PyTorch baselines.
    for (llama_token id : o.ids) decode_one(s.ctx, id);
    Generation g;
    double decode_total = 0, sample_total = 0;
    for (int i = 0; i < o.new_tokens; ++i) {
        const auto t0 = clock_type::now();
        const llama_token next = llama_sampler_sample(greedy, s.ctx, -1);
        sample_total += elapsed_us(t0);
        g.top_logits.push_back(s.top().logit);
        g.tokens.push_back(next);
        const llama_token fed = o.forced.empty() ? next : o.forced.at(i);
        const auto t1 = clock_type::now();
        decode_one(s.ctx, fed);
        decode_total += elapsed_us(t1);
    }
    g.decode_us_per_token = decode_total / o.new_tokens;
    g.sample_us_per_token = sample_total / o.new_tokens;
    return g;
}

void run_greedy(const Options & o) {
    if (o.ids.empty()) throw std::runtime_error("--ids is required");
    if (!o.forced.empty() && (int) o.forced.size() < o.new_tokens) throw std::runtime_error("--force needs new-tokens ids");
    Session s(o, (int) o.ids.size() + o.new_tokens + 1);
    llama_sampler * greedy = llama_sampler_init_greedy();
    Generation first;
    std::vector<double> decode_us, sample_us;
    for (int r = -o.warmup; r < o.rounds; ++r) {
        Generation g = generate(s, o, greedy);
        if (r == -o.warmup) first = g;
        else if (g.tokens != first.tokens) throw std::runtime_error("generation differs across rounds");
        if (r >= 0) {
            decode_us.push_back(g.decode_us_per_token);
            sample_us.push_back(g.sample_us_per_token);
        }
    }
    std::printf("{\"mode\":\"greedy\",\"n_vocab\":%d,\"tokens\":[", s.n_vocab);
    for (size_t i = 0; i < first.tokens.size(); ++i) std::printf("%s%d", i ? "," : "", first.tokens[i]);
    std::printf("],");
    print_array("top_logits", first.top_logits);
    std::printf(",");
    print_array("decode_us_per_token", decode_us);
    std::printf(",");
    print_array("sample_us_per_token", sample_us);
    std::printf("}\n");
    llama_sampler_free(greedy);
}

// Rounds of `iterations` decodes at one fixed position. Each call removes the
// previous token at that position first, so the context length stays constant.
std::vector<double> time_position(Session & s, int position, llama_token token, const Options & o,
                                  llama_sampler * sampler) {
    llama_memory_t mem = llama_get_memory(s.ctx);
    auto call = [&] {
        llama_memory_seq_rm(mem, 0, position, -1);
        decode_one(s.ctx, token);
        if (sampler) (void) llama_sampler_sample(sampler, s.ctx, -1);
    };
    for (int i = 0; i < o.warmup; ++i) call();
    std::vector<double> samples;
    for (int r = 0; r < o.rounds; ++r) {
        const auto t0 = clock_type::now();
        for (int i = 0; i < o.iterations; ++i) call();
        samples.push_back(elapsed_us(t0) / o.iterations);
    }
    return samples;
}

void run_decode(const Options & o) {
    const int max_position = *std::max_element(o.positions.begin(), o.positions.end());
    Session s(o, max_position + 2);
    llama_sampler * greedy = llama_sampler_init_greedy();
    llama_memory_t mem = llama_get_memory(s.ctx);
    std::printf("{\"mode\":\"decode\",\"n_vocab\":%d,\"rows\":[", s.n_vocab);
    bool first = true;
    for (int position : o.positions) {
        llama_memory_clear(mem, true);
        // Prefix setup is untimed and fed one token at a time.
        for (int p = 0; p < position; ++p) decode_one(s.ctx, o.token);
        const std::vector<double> none = time_position(s, position, o.token, o, nullptr);
        const std::vector<double> sampled = time_position(s, position, o.token, o, greedy);
        // One more decode to read the logits at this position.
        llama_memory_seq_rm(mem, 0, position, -1);
        decode_one(s.ctx, o.token);
        const Top top = s.top();
        std::printf("%s{\"position\":%d,\"context_length\":%d,\"next_token\":%d,\"top_logit\":%.6f,", first ? "" : ",",
                    position, position + 1, top.id, top.logit);
        print_array("none_samples_us", none);
        std::printf(",");
        print_array("greedy_samples_us", sampled);
        std::printf("}");
        first = false;
    }
    std::printf("]}\n");
    llama_sampler_free(greedy);
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const Options o = parse(argc, argv);
        if (o.mode == "greedy") run_greedy(o);
        else if (o.mode == "decode") run_decode(o);
        else throw std::runtime_error("mode must be greedy or decode");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "rt_probe: %s\n", e.what());
        return 1;
    }
}
