#include "llama_context.h"

#include <godot_cpp/classes/os.hpp>
#include <godot_cpp/classes/time.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include <llama.h>
#include <algorithm>
#include <vector>

using namespace godot;

void LlamaContext::_bind_methods() {
    ClassDB::bind_method(D_METHOD("create", "model", "params"), &LlamaContext::create, DEFVAL(Dictionary()));
    ClassDB::bind_method(D_METHOD("reset"), &LlamaContext::reset);
    ClassDB::bind_method(D_METHOD("set_prompt", "prompt"), &LlamaContext::set_prompt);
    ClassDB::bind_method(D_METHOD("generate", "max_tokens", "params"), &LlamaContext::generate, DEFVAL(128), DEFVAL(Dictionary()));
    ClassDB::bind_method(D_METHOD("generate_stream", "max_tokens", "params"), &LlamaContext::generate_stream, DEFVAL(128), DEFVAL(Dictionary()));
    ClassDB::bind_method(D_METHOD("cancel"), &LlamaContext::cancel);
    ClassDB::bind_method(D_METHOD("get_stats"), &LlamaContext::get_stats);
    ClassDB::bind_method(D_METHOD("get_model"), &LlamaContext::get_model);
    ClassDB::bind_method(D_METHOD("get_prompt"), &LlamaContext::get_prompt);
    ClassDB::bind_method(D_METHOD("is_initialized"), &LlamaContext::is_initialized);

    ADD_SIGNAL(MethodInfo("token_generated", PropertyInfo(Variant::STRING, "token_text"), PropertyInfo(Variant::INT, "token_id")));
    ADD_SIGNAL(MethodInfo("generation_finished", PropertyInfo(Variant::STRING, "full_text")));
    ADD_SIGNAL(MethodInfo("generation_error", PropertyInfo(Variant::STRING, "message")));
}

LlamaContext::~LlamaContext() {
    if (native_sampler != nullptr) {
        llama_sampler_free(native_sampler);
        native_sampler = nullptr;
    }
    if (native_context != nullptr) {
        llama_free(native_context);
        native_context = nullptr;
    }
}

bool LlamaContext::_is_ready() const {
    return model.is_valid() && model->is_loaded() && native_context != nullptr && native_sampler != nullptr;
}

void LlamaContext::_emit_error(const String &p_message) const {
    UtilityFunctions::push_error("godot_llama: ", p_message);
    const_cast<LlamaContext *>(this)->emit_signal("generation_error", p_message);
}

bool LlamaContext::_decode_tokens(const std::vector<int32_t> &p_tokens) {
    if (p_tokens.empty()) {
        return true;
    }

    std::vector<llama_token> tokens;
    tokens.reserve(p_tokens.size());
    for (size_t i = 0; i < p_tokens.size(); i++) {
        tokens.push_back(static_cast<llama_token>(p_tokens[i]));
    }

    llama_batch batch = llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()));
    int32_t rc = llama_decode(native_context, batch);
    return rc == 0;
}

String LlamaContext::_token_to_piece(int32_t p_token) const {
    if (!_is_ready()) {
        return "";
    }

    const llama_vocab *vocab = model->get_vocab();
    std::vector<char> piece(64, '\0');
    int32_t rc = llama_token_to_piece(vocab, static_cast<llama_token>(p_token), piece.data(), static_cast<int32_t>(piece.size()), 0, true);
    if (rc < 0) {
        piece.resize(-rc + 1);
        rc = llama_token_to_piece(vocab, static_cast<llama_token>(p_token), piece.data(), static_cast<int32_t>(piece.size()), 0, true);
    }
    if (rc <= 0) {
        return "";
    }
    return String::utf8(piece.data(), rc);
}

Error LlamaContext::create(const Ref<LlamaModel> &p_model, const Dictionary &p_params) {
    if (native_sampler != nullptr) {
        llama_sampler_free(native_sampler);
        native_sampler = nullptr;
    }
    if (native_context != nullptr) {
        llama_free(native_context);
        native_context = nullptr;
    }

    model = p_model;
    if (model.is_null() || !model->is_loaded()) {
        return ERR_UNCONFIGURED;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 2048;
    cparams.n_batch = 512;
    cparams.n_ubatch = 512;
    cparams.n_threads = std::max(1, OS::get_singleton()->get_processor_count() - 1);
    cparams.n_threads_batch = cparams.n_threads;

    if (p_params.has("n_ctx")) {
        cparams.n_ctx = static_cast<uint32_t>(int64_t(p_params["n_ctx"]));
    }
    if (p_params.has("n_batch")) {
        cparams.n_batch = static_cast<uint32_t>(int64_t(p_params["n_batch"]));
        cparams.n_ubatch = cparams.n_batch;
    }
    if (p_params.has("threads")) {
        cparams.n_threads = static_cast<int32_t>(int64_t(p_params["threads"]));
    }
    if (p_params.has("threads_batch")) {
        cparams.n_threads_batch = static_cast<int32_t>(int64_t(p_params["threads_batch"]));
    }

    native_context = llama_init_from_model(const_cast<llama_model *>(model->get_native_model()), cparams);
    if (native_context == nullptr) {
        return ERR_CANT_CREATE;
    }

    llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    native_sampler = llama_sampler_chain_init(chain_params);
    llama_sampler_chain_add(native_sampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(native_sampler, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(native_sampler, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(native_sampler, llama_sampler_init_dist(static_cast<uint32_t>(Time::get_singleton()->get_unix_time_from_system())));

    return OK;
}

void LlamaContext::reset() {
    if (native_context != nullptr) {
        llama_memory_clear(llama_get_memory(native_context), true);
        llama_perf_context_reset(native_context);
    }
    if (native_sampler != nullptr) {
        llama_sampler_reset(native_sampler);
    }
}

void LlamaContext::set_prompt(const String &p_prompt) {
    prompt = p_prompt;
}

String LlamaContext::_generate_internal(int p_max_tokens, const Dictionary &p_params, bool p_streaming) {
    if (!_is_ready()) {
        _emit_error("Context is not initialized. Call create() with a loaded model.");
        return "";
    }

    if (prompt.is_empty()) {
        _emit_error("Prompt is empty. Call set_prompt() first.");
        return "";
    }

    int max_tokens = p_max_tokens;
    if (p_params.has("max_tokens")) {
        max_tokens = static_cast<int>(int64_t(p_params["max_tokens"]));
    }
    if (max_tokens <= 0) {
        return "";
    }

    float temperature = 0.7f;
    float top_p = 0.9f;
    int32_t top_k = 40;
    uint32_t seed = static_cast<uint32_t>(Time::get_singleton()->get_unix_time_from_system());
    if (p_params.has("temperature")) {
        temperature = static_cast<float>(double(p_params["temperature"]));
    }
    if (p_params.has("top_p")) {
        top_p = static_cast<float>(double(p_params["top_p"]));
    }
    if (p_params.has("top_k")) {
        top_k = static_cast<int32_t>(int64_t(p_params["top_k"]));
    }
    if (p_params.has("seed")) {
        seed = static_cast<uint32_t>(int64_t(p_params["seed"]));
    }

    if (native_sampler != nullptr) {
        llama_sampler_free(native_sampler);
    }
    llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    native_sampler = llama_sampler_chain_init(chain_params);
    llama_sampler_chain_add(native_sampler, llama_sampler_init_top_k(top_k));
    llama_sampler_chain_add(native_sampler, llama_sampler_init_top_p(top_p, 1));
    llama_sampler_chain_add(native_sampler, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(native_sampler, llama_sampler_init_dist(seed));

    cancel_requested = false;

    PackedInt32Array prompt_tokens_gd = model->tokenize(prompt, true);
    if (prompt_tokens_gd.is_empty()) {
        _emit_error("Tokenization failed for prompt.");
        return "";
    }

    std::vector<int32_t> prompt_tokens;
    prompt_tokens.reserve(prompt_tokens_gd.size());
    for (int i = 0; i < prompt_tokens_gd.size(); i++) {
        prompt_tokens.push_back(prompt_tokens_gd[i]);
    }

    if (!_decode_tokens(prompt_tokens)) {
        _emit_error("llama_decode failed while processing prompt.");
        return "";
    }

    String full_text;
    const llama_vocab *vocab = model->get_vocab();
    for (int i = 0; i < max_tokens; i++) {
        if (cancel_requested) {
            break;
        }

        llama_token token = llama_sampler_sample(native_sampler, native_context, -1);
        if (llama_vocab_is_eog(vocab, token)) {
            break;
        }

        const String token_text = _token_to_piece(token);
        full_text += token_text;

        if (p_streaming) {
            emit_signal("token_generated", token_text, static_cast<int64_t>(token));
        }

        llama_sampler_accept(native_sampler, token);
        std::vector<int32_t> next_token = { static_cast<int32_t>(token) };
        if (!_decode_tokens(next_token)) {
            _emit_error("llama_decode failed while generating tokens.");
            return full_text;
        }
    }

    emit_signal("generation_finished", full_text);
    return full_text;
}

String LlamaContext::generate(int p_max_tokens, const Dictionary &p_params) {
    return _generate_internal(p_max_tokens, p_params, false);
}

void LlamaContext::generate_stream(int p_max_tokens, const Dictionary &p_params) {
    _generate_internal(p_max_tokens, p_params, true);
}

void LlamaContext::cancel() {
    cancel_requested = true;
}

Dictionary LlamaContext::get_stats() const {
    Dictionary stats;
    if (!_is_ready()) {
        return stats;
    }

    const llama_perf_context_data perf = llama_perf_context(native_context);
    stats["t_start_ms"] = perf.t_start_ms;
    stats["t_load_ms"] = perf.t_load_ms;
    stats["t_p_eval_ms"] = perf.t_p_eval_ms;
    stats["t_eval_ms"] = perf.t_eval_ms;
    stats["n_p_eval"] = perf.n_p_eval;
    stats["n_eval"] = perf.n_eval;
    stats["n_reused"] = perf.n_reused;
    stats["n_ctx"] = static_cast<int64_t>(llama_n_ctx(native_context));
    return stats;
}

Ref<LlamaModel> LlamaContext::get_model() const {
    return model;
}

String LlamaContext::get_prompt() const {
    return prompt;
}

bool LlamaContext::is_initialized() const {
    return native_context != nullptr;
}
