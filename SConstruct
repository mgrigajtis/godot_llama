#!/usr/bin/env python

import os
import shutil
import sys

sys.path.append(Dir("godot-cpp").srcnode().abspath)

env = SConscript("godot-cpp/SConstruct")
llama_build_dir = os.environ.get("LLAMA_CPP_BUILD_DIR", "third_party/llama.cpp/build")

def _env_flag(name):
    value = os.environ.get(name, "").strip().lower()
    return value in ("1", "true", "yes", "on")

def _strip_link_flag(flag):
    if "LINKFLAGS" not in env:
        return
    while flag in env["LINKFLAGS"]:
        env["LINKFLAGS"].remove(flag)

if env["platform"] == "linux" and env.get("use_static_cpp"):
    print("warning: use_static_cpp=yes can crash when loaded by Godot on Linux; forcing dynamic libstdc++/libgcc")
    _strip_link_flag("-static-libgcc")
    _strip_link_flag("-static-libstdc++")

env.Append(CPPPATH=[
    "src/",
    "third_party/llama.cpp/include",
    "third_party/llama.cpp/ggml/include",
])

link_static_llama = _env_flag("LLAMA_CPP_LINK_STATIC")
if env["platform"] == "linux" and "LLAMA_CPP_LINK_STATIC" not in os.environ:
    link_static_llama = True
if env["platform"] == "windows" and link_static_llama:
    print("warning: LLAMA_CPP_LINK_STATIC is not supported on Windows; ignoring")
    link_static_llama = False

openmp_env_set = "LLAMA_CPP_OPENMP" in os.environ
link_openmp = _env_flag("LLAMA_CPP_OPENMP")
if env["platform"] == "linux" and link_static_llama and not openmp_env_set:
    link_openmp = True

if env["platform"] == "windows":
    env.Append(LIBPATH=[
        os.path.join(llama_build_dir, "src", "Release"),
        os.path.join(llama_build_dir, "ggml", "src", "Release"),
        os.path.join(llama_build_dir, "common", "Release"),
    ])
    env.Append(LIBS=["advapi32"])
else:
    env.Append(LIBPATH=[
        os.path.join(llama_build_dir, "src"),
        os.path.join(llama_build_dir, "ggml", "src"),
        os.path.join(llama_build_dir, "common"),
    ])

static_libs = [
    os.path.join(llama_build_dir, "src", "libllama.a"),
    os.path.join(llama_build_dir, "ggml", "src", "libggml.a"),
    os.path.join(llama_build_dir, "ggml", "src", "libggml-base.a"),
    os.path.join(llama_build_dir, "ggml", "src", "libggml-cpu.a"),
]
static_lib_nodes = [env.File(path) for path in static_libs]

if link_static_llama and env["platform"] != "windows":
    missing = [lib for lib in static_libs if not os.path.isfile(lib)]
    if missing:
        print("warning: requested static llama.cpp link, missing:")
        for lib in missing:
            print("  -", lib)
        print("warning: falling back to shared libraries")
        if env["platform"] == "linux" and not openmp_env_set:
            link_openmp = False
        env.Append(LIBS=["llama", "ggml", "ggml-cpu", "ggml-base"])
    else:
        if env["platform"] == "linux":
            env.Append(LINKFLAGS=["-Wl,--start-group"])
            env.Append(LIBS=static_lib_nodes)
            env.Append(LINKFLAGS=["-Wl,--end-group"])
        else:
            env.Append(LIBS=static_lib_nodes)
else:
    env.Append(LIBS=["llama", "ggml", "ggml-cpu", "ggml-base"])

if env["platform"] == "linux":
    env.Append(LIBS=["pthread", "dl", "m"])
    if link_openmp:
        env.Append(LINKFLAGS=["-fopenmp"])

sources = Glob("src/*.cpp")

if env["platform"] == "macos":
    library = env.SharedLibrary(
        "addons/godot_llama/bin/libgodot_llama.{}.{}.framework/libgodot_llama.{}.{}".format(
            env["platform"], env["target"], env["platform"], env["target"]
        ),
        source=sources,
    )
elif env["platform"] == "ios":
    if env["ios_simulator"]:
        library = env.StaticLibrary(
            "addons/godot_llama/bin/libgodot_llama.{}.{}.simulator.a".format(env["platform"], env["target"]),
            source=sources,
        )
    else:
        library = env.StaticLibrary(
            "addons/godot_llama/bin/libgodot_llama.{}.{}.a".format(env["platform"], env["target"]),
            source=sources,
        )
else:
    library = env.SharedLibrary(
        "addons/godot_llama/bin/libgodot_llama{}{}".format(env["suffix"], env["SHLIBSUFFIX"]),
        source=sources,
    )

if env["platform"] == "windows" and not link_static_llama:
    def _sync_runtime_dlls(target, source, env):
        runtime_dir = os.path.join(llama_build_dir, "bin", "Release")
        required = ["llama.dll", "ggml.dll", "ggml-base.dll", "ggml-cpu.dll", "mtmd.dll"]
        destinations = ["addons/godot_llama/bin", "."]

        for dll in required:
            src = os.path.join(runtime_dir, dll)
            if not os.path.isfile(src):
                print("warning: missing runtime DLL:", src)
                continue

            for dst_dir in destinations:
                os.makedirs(dst_dir, exist_ok=True)
                shutil.copy2(src, os.path.join(dst_dir, dll))

        with open(str(target[0]), "w", encoding="utf-8") as stamp:
            stamp.write("runtime sync complete\n")
        return 0

    runtime_sync = env.Command(
        "addons/godot_llama/bin/.runtime_sync_{}.stamp".format(env["target"]),
        library,
        _sync_runtime_dlls,
    )
    env.NoCache(runtime_sync)
    Default(runtime_sync)
else:
    Default(library)

env.NoCache(library)
