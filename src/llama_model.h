#ifndef GODOT_LLAMA_MODEL_H
#define GODOT_LLAMA_MODEL_H

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <godot_cpp/variant/string.hpp>

struct llama_model;
struct llama_vocab;

namespace godot {

class LlamaModel : public RefCounted {
    GDCLASS(LlamaModel, RefCounted);

private:
    struct llama_model *native_model = nullptr;
    const struct llama_vocab *vocab = nullptr;
    String model_path;

    bool _load_tokenize_internal(const String &p_text, bool p_add_bos, PackedInt32Array &r_tokens) const;
    static String _globalize_path(const String &p_path);

protected:
    static void _bind_methods();

public:
    ~LlamaModel();

    Error load(const String &p_model_path, const Dictionary &p_params = Dictionary());
    void unload();
    bool is_loaded() const;
    String get_model_path() const;
    PackedInt32Array tokenize(const String &p_text, bool p_add_bos = true) const;
    String detokenize(const PackedInt32Array &p_tokens) const;
    int get_vocab_size() const;
    Dictionary get_metadata() const;

    const struct llama_model *get_native_model() const;
    const struct llama_vocab *get_vocab() const;
};

} // namespace godot

#endif
