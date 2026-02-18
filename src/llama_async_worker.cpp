#include "llama_async_worker.h"

#include <godot_cpp/core/class_db.hpp>

using namespace godot;

void LlamaAsyncWorker::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_context", "context"), &LlamaAsyncWorker::set_context);
    ClassDB::bind_method(D_METHOD("get_context"), &LlamaAsyncWorker::get_context);
    ClassDB::bind_method(D_METHOD("start_generation", "prompt", "max_tokens"), &LlamaAsyncWorker::start_generation, DEFVAL(128));
    ClassDB::bind_method(D_METHOD("is_busy"), &LlamaAsyncWorker::is_busy);
    ClassDB::bind_method(D_METHOD("get_latest_output"), &LlamaAsyncWorker::get_latest_output);

    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "context", PROPERTY_HINT_RESOURCE_TYPE, "LlamaContext"), "set_context", "get_context");
}

void LlamaAsyncWorker::_thread_entry() {
    if (context.is_valid()) {
        context->set_prompt(pending_prompt);
        latest_output = context->generate(pending_max_tokens);
    } else {
        latest_output = "";
    }
    busy = false;
}

void LlamaAsyncWorker::set_context(const Ref<LlamaContext> &p_context) {
    context = p_context;
}

Ref<LlamaContext> LlamaAsyncWorker::get_context() const {
    return context;
}

Error LlamaAsyncWorker::start_generation(const String &p_prompt, int p_max_tokens) {
    if (busy) {
        return ERR_BUSY;
    }
    if (context.is_null()) {
        return ERR_UNCONFIGURED;
    }

    if (thread.is_valid() && thread->is_started()) {
        thread->wait_to_finish();
    }

    pending_prompt = p_prompt;
    pending_max_tokens = p_max_tokens;
    busy = true;

    if (thread.is_null()) {
        thread.instantiate();
    }

    thread->start(callable_mp(this, &LlamaAsyncWorker::_thread_entry));
    return OK;
}

bool LlamaAsyncWorker::is_busy() const {
    return busy;
}

String LlamaAsyncWorker::get_latest_output() const {
    return latest_output;
}
