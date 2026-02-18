#ifndef GODOT_LLAMA_CONTEXT_H
#define GODOT_LLAMA_CONTEXT_H

#include "llama_model.h"

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/string.hpp>
#include <vector>

struct llama_context;
struct llama_sampler;

namespace godot {

class LlamaContext : public RefCounted {
    GDCLASS(LlamaContext, RefCounted);

private:
    struct llama_context *native_context = nullptr;
    struct llama_sampler *native_sampler = nullptr;

    Ref<LlamaModel> model;
    String prompt;
    bool cancel_requested = false;

    bool _is_ready() const;
    void _emit_error(const String &p_message) const;
    String _generate_internal(int p_max_tokens, const Dictionary &p_params, bool p_streaming);
    bool _decode_tokens(const std::vector<int32_t> &p_tokens);
    String _token_to_piece(int32_t p_token) const;

protected:
    static void _bind_methods();

public:
    ~LlamaContext();

    Error create(const Ref<LlamaModel> &p_model, const Dictionary &p_params = Dictionary());
    void reset();
    void set_prompt(const String &p_prompt);
    String generate(int p_max_tokens = 128, const Dictionary &p_params = Dictionary());
    void generate_stream(int p_max_tokens = 128, const Dictionary &p_params = Dictionary());
    void cancel();
    Dictionary get_stats() const;

    Ref<LlamaModel> get_model() const;
    String get_prompt() const;
    bool is_initialized() const;
};

} // namespace godot

#endif
