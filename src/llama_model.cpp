#include "llama_model.h"

#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/classes/project_settings.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/error_macros.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include <llama.h>
#include <climits>
#include <vector>

using namespace godot;

String LlamaModel::_globalize_path(const String &p_path) {
    if (p_path.begins_with("res://") || p_path.begins_with("user://")) {
        return ProjectSettings::get_singleton()->globalize_path(p_path);
    }
    return p_path;
}

bool LlamaModel::_load_tokenize_internal(const String &p_text, bool p_add_bos, PackedInt32Array &r_tokens) const {
    if (!is_loaded()) {
        return false;
    }

    CharString text_utf8 = p_text.utf8();
    const int32_t text_len = static_cast<int32_t>(text_utf8.length());

    int32_t token_count = llama_tokenize(vocab, text_utf8.get_data(), text_len, nullptr, 0, p_add_bos, false);
    if (token_count == INT32_MIN) {
        return false;
    }

    if (token_count < 0) {
        token_count = -token_count;
    }
    if (token_count <= 0) {
        return true;
    }

    std::vector<llama_token> tokens(token_count);
    int32_t rc = llama_tokenize(vocab, text_utf8.get_data(), text_len, tokens.data(), token_count, p_add_bos, false);
    if (rc < 0) {
        return false;
    }

    for (int32_t i = 0; i < rc; i++) {
        r_tokens.append(tokens[i]);
    }
    return true;
}

void LlamaModel::_bind_methods() {
    ClassDB::bind_method(D_METHOD("load", "model_path", "params"), &LlamaModel::load, DEFVAL(Dictionary()));
    ClassDB::bind_method(D_METHOD("unload"), &LlamaModel::unload);
    ClassDB::bind_method(D_METHOD("is_loaded"), &LlamaModel::is_loaded);
    ClassDB::bind_method(D_METHOD("get_model_path"), &LlamaModel::get_model_path);
    ClassDB::bind_method(D_METHOD("tokenize", "text", "add_bos"), &LlamaModel::tokenize, DEFVAL(true));
    ClassDB::bind_method(D_METHOD("detokenize", "tokens"), &LlamaModel::detokenize);
    ClassDB::bind_method(D_METHOD("get_vocab_size"), &LlamaModel::get_vocab_size);
    ClassDB::bind_method(D_METHOD("get_metadata"), &LlamaModel::get_metadata);
}

LlamaModel::~LlamaModel() {
    unload();
}

Error LlamaModel::load(const String &p_model_path, const Dictionary &p_params) {
    unload();

    const String global_path = _globalize_path(p_model_path);
    if (!FileAccess::file_exists(global_path)) {
        return ERR_FILE_NOT_FOUND;
    }

    llama_model_params mparams = llama_model_default_params();
    if (p_params.has("n_gpu_layers")) {
        mparams.n_gpu_layers = static_cast<int32_t>(int64_t(p_params["n_gpu_layers"]));
    }
    if (p_params.has("use_mmap")) {
        mparams.use_mmap = bool(p_params["use_mmap"]);
    }
    if (p_params.has("use_mlock")) {
        mparams.use_mlock = bool(p_params["use_mlock"]);
    }
    if (p_params.has("vocab_only")) {
        mparams.vocab_only = bool(p_params["vocab_only"]);
    }
    if (p_params.has("check_tensors")) {
        mparams.check_tensors = bool(p_params["check_tensors"]);
    }

    CharString path_utf8 = global_path.utf8();
    native_model = llama_model_load_from_file(path_utf8.get_data(), mparams);
    if (native_model == nullptr) {
        UtilityFunctions::push_error("godot_llama: failed to load model: ", global_path);
        return ERR_CANT_OPEN;
    }

    vocab = llama_model_get_vocab(native_model);
    model_path = p_model_path;
    return OK;
}

void LlamaModel::unload() {
    if (native_model != nullptr) {
        llama_model_free(native_model);
        native_model = nullptr;
    }
    vocab = nullptr;
    model_path = "";
}

bool LlamaModel::is_loaded() const {
    return native_model != nullptr;
}

String LlamaModel::get_model_path() const {
    return model_path;
}

PackedInt32Array LlamaModel::tokenize(const String &p_text, bool p_add_bos) const {
    PackedInt32Array tokens;
    if (!_load_tokenize_internal(p_text, p_add_bos, tokens)) {
        UtilityFunctions::push_error("godot_llama: tokenize failed");
    }
    return tokens;
}

String LlamaModel::detokenize(const PackedInt32Array &p_tokens) const {
    if (!is_loaded()) {
        return "";
    }

    std::vector<llama_token> tokens;
    tokens.reserve(p_tokens.size());
    for (int i = 0; i < p_tokens.size(); i++) {
        tokens.push_back(static_cast<llama_token>(p_tokens[i]));
    }

    int32_t text_len = llama_detokenize(vocab, tokens.data(), static_cast<int32_t>(tokens.size()), nullptr, 0, true, false);
    if (text_len < 0) {
        text_len = -text_len;
    }
    if (text_len <= 0) {
        return "";
    }

    std::vector<char> out(text_len + 1, '\0');
    int32_t rc = llama_detokenize(vocab, tokens.data(), static_cast<int32_t>(tokens.size()), out.data(), text_len, true, false);
    if (rc < 0) {
        return "";
    }

    return String::utf8(out.data(), rc);
}

int LlamaModel::get_vocab_size() const {
    if (!is_loaded()) {
        return 0;
    }
    return llama_vocab_n_tokens(vocab);
}

Dictionary LlamaModel::get_metadata() const {
    Dictionary metadata;
    if (!is_loaded()) {
        return metadata;
    }

    const int32_t count = llama_model_meta_count(native_model);
    for (int32_t i = 0; i < count; i++) {
        std::vector<char> key_buf(512, '\0');
        std::vector<char> val_buf(2048, '\0');

        int32_t key_len = llama_model_meta_key_by_index(native_model, i, key_buf.data(), static_cast<size_t>(key_buf.size()));
        int32_t val_len = llama_model_meta_val_str_by_index(native_model, i, val_buf.data(), static_cast<size_t>(val_buf.size()));
        if (key_len <= 0 || val_len < 0) {
            continue;
        }

        metadata[String::utf8(key_buf.data())] = String::utf8(val_buf.data());
    }
    return metadata;
}

const struct llama_model *LlamaModel::get_native_model() const {
    return native_model;
}

const struct llama_vocab *LlamaModel::get_vocab() const {
    return vocab;
}
