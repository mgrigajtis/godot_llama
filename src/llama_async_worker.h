#ifndef GODOT_LLAMA_ASYNC_WORKER_H
#define GODOT_LLAMA_ASYNC_WORKER_H

#include "llama_context.h"

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/classes/thread.hpp>
#include <godot_cpp/variant/string.hpp>

namespace godot {

class LlamaAsyncWorker : public RefCounted {
    GDCLASS(LlamaAsyncWorker, RefCounted);

private:
    Ref<LlamaContext> context;
    Ref<Thread> thread;
    bool busy = false;
    String pending_prompt;
    int pending_max_tokens = 128;
    String latest_output;

    void _thread_entry();

protected:
    static void _bind_methods();

public:
    void set_context(const Ref<LlamaContext> &p_context);
    Ref<LlamaContext> get_context() const;

    Error start_generation(const String &p_prompt, int p_max_tokens = 128);
    bool is_busy() const;
    String get_latest_output() const;
};

} // namespace godot

#endif
