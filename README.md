# godot_llama

Godot 4.x GDExtension wrapper for `llama.cpp`.

## Status

Implemented:
- `godot-cpp` and `third_party/llama.cpp` submodules
- GDExtension build setup via `SConstruct`
- Native classes registered to Godot:
  - `LlamaModel`
  - `LlamaSampler`
  - `LlamaContext`
  - `LlamaAsyncWorker`
- Addon manifest and GDScript facade in `addons/godot_llama/`
- `LlamaModel` now uses real `llama.cpp` model loading, tokenization, detokenization, vocab size, and metadata APIs.
- `LlamaContext` now uses real `llama.cpp` context creation, sampling, synchronous generation, streaming generation signals, cancellation, and perf stats.
- Runnable demo scene with GUI controls for:
  - GGUF file selection (file picker)
  - model/context creation
  - prompt/system/world-state editing
  - generation settings and streaming
  - short conversation memory and memory reset

## Submodules

```bash
git submodule update --init --recursive
```

`godot-cpp` is set to branch `4.4`.

## Build (GDExtension)

Prereqs:
- Python 3.8+
- SCons 4+
- C++ toolchain for your platform

Build `llama.cpp` shared libs (Windows):

```powershell
cmake -S third_party/llama.cpp -B third_party/llama.cpp/build_shared -DGGML_SHARED=ON -DBUILD_SHARED_LIBS=ON
cmake --build third_party/llama.cpp/build_shared --config Release
```

Build extension:

```bash
$env:LLAMA_CPP_BUILD_DIR="third_party/llama.cpp/build_shared"
scons target=template_debug platform=windows use_static_cpp=no -j8
scons target=template_release platform=windows use_static_cpp=no -j8
```

The SCons build automatically copies required runtime DLLs (`llama.dll`, `ggml.dll`, `ggml-base.dll`, `ggml-cpu.dll`, `mtmd.dll`) to:
- `addons/godot_llama/bin/`
- repo root (`./`) for editor-time Windows DLL resolution.

Build `llama.cpp` static libs (Linux):

```bash
cmake -S third_party/llama.cpp -B third_party/llama.cpp/build \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON
cmake --build third_party/llama.cpp/build --config Release
```

Build extension (Linux):

```bash
scons target=template_debug platform=linux use_static_cpp=no -j8
scons target=template_release platform=linux use_static_cpp=no -j8
```

Artifacts are emitted under:

`addons/godot_llama/bin/`

Optional override for non-default llama build output:
- `LLAMA_CPP_BUILD_DIR` (example: `third_party/llama.cpp/build_shared`)
- `LLAMA_CPP_LINK_STATIC=1` to force static llama.cpp linking (Linux defaults to static when the variable is unset)
- `LLAMA_CPP_OPENMP=0` to skip linking OpenMP on Linux (use this if you built llama.cpp with `-DGGML_OPENMP=OFF`)
- `use_static_cpp=no` is required on Linux to avoid crashes from mixing static libstdc++ with Godot's runtime

## Godot addon files

- `addons/godot_llama/godot_llama.gdextension`
- `addons/godot_llama/godot_llama.gd`

## Demo scene

This repo intentionally does not include a `project.godot`.

Demo files are included inside the addon:
- `addons/godot_llama/demo/demo.tscn`
- `addons/godot_llama/demo/demo.gd`

To run the demo:
1. Open your own Godot 4.4+ project.
2. Copy or include `addons/godot_llama/` in that project.
3. Open `res://addons/godot_llama/demo/demo.tscn`.
4. Click `Select Model...` and choose a `.gguf` file.
5. Click `Load Model`, then `Create Context`.
6. Set `System prompt`, `World state`, and `history_turns` as needed.
7. Enter a player prompt and click `Generate`.

Recommended starting settings for NPC dialog:
- `n_ctx`: `1024` or `2048`
- `temperature`: `0.6` to `0.8`
- `top_p`: `0.85` to `0.95`
- `max_tokens`: `80` to `160`
