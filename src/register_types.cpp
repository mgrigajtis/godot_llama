#include "register_types.h"

#include "llama_async_worker.h"
#include "llama_context.h"
#include "llama_model.h"
#include "llama_sampler.h"

#include <godot_cpp/core/defs.hpp>
#include <godot_cpp/godot.hpp>

#include <llama.h>

using namespace godot;

void initialize_godot_llama_module(ModuleInitializationLevel p_level) {
    if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
        return;
    }

    llama_backend_init();

    ClassDB::register_class<LlamaModel>();
    ClassDB::register_class<LlamaSampler>();
    ClassDB::register_class<LlamaContext>();
    ClassDB::register_class<LlamaAsyncWorker>();
}

void uninitialize_godot_llama_module(ModuleInitializationLevel p_level) {
    if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
        return;
    }

    llama_backend_free();
}

extern "C" {
GDExtensionBool GDE_EXPORT godot_llama_library_init(
        GDExtensionInterfaceGetProcAddress p_get_proc_address,
        const GDExtensionClassLibraryPtr p_library,
        GDExtensionInitialization *r_initialization) {
    GDExtensionBinding::InitObject init_obj(p_get_proc_address, p_library, r_initialization);
    init_obj.register_initializer(initialize_godot_llama_module);
    init_obj.register_terminator(uninitialize_godot_llama_module);
    init_obj.set_minimum_library_initialization_level(MODULE_INITIALIZATION_LEVEL_SCENE);
    return init_obj.init();
}
}
