#!/usr/bin/env python

import os
import shutil
import sys

sys.path.append(Dir("godot-cpp").srcnode().abspath)

env = SConscript("godot-cpp/SConstruct")
llama_build_dir = os.environ.get("LLAMA_CPP_BUILD_DIR", "third_party/llama.cpp/build")

env.Append(CPPPATH=[
    "src/",
    "third_party/llama.cpp/include",
    "third_party/llama.cpp/ggml/include",
])

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

env.Append(LIBS=["llama", "ggml", "ggml-cpu", "ggml-base"])

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

if env["platform"] == "windows":
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
