# Handoff — Resume Builder Dev Environment Setup

## Status: ✅ BUILDS & RUNS

The app compiles (585 crates) and launches. To start it: **`.\run.ps1`** (or
`npm run tauri dev` with the environment set up as below).

> **Local LLM = Ollama.** The embedded `llama-cpp-2`/GGUF path was replaced by an
> external **Ollama** server over HTTP (model `qwen3.5:9b`). The `llama-cpp-2`
> dependency and `local-llm` feature were removed, so **cmake and libclang are no
> longer required to build** (they remain installed but unused). See
> `src-tauri/src/llm/mod.rs::generate_ollama`. Reasoning models route output to a
> separate `thinking` field, so requests send `"think": false` to get the answer
> in `response`.

## What This Is

A Tauri 2 desktop app (Rust backend + React/TypeScript frontend) that manages a
master resume database, generates AI-powered cover letters via RAG + Gemini API,
and compiles LaTeX resumes to PDF using Tectonic.

---

## Toolchain (all installed without admin rights)

| Tool | Version | Location |
|---|---|---|
| Node.js (portable zip) | v20.18.3 | `%LOCALAPPDATA%\nodejs\node-v20.18.3-win-x64\` |
| Rust (via rustup) | 1.96.0 | `%USERPROFILE%\.cargo\bin\` |
| MinGW-w64 GCC (portable) | 14.2.0 | `%LOCALAPPDATA%\mingw64\mingw64\bin\` (unused — MSVC target) |
| **VS Build Tools 2022** (MSVC `link.exe`, v143 toolset) | 14.44.35207 | `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\` |
| **libclang** (from PyPI `libclang` wheel) | 18.1.1 | `C:\Users\smart\libclang-bin\libclang.dll` |
| **CMake** (portable zip) | 4.3.3 | `C:\Users\smart\cmake\cmake-4.3.3-windows-x86_64\bin\` |

**Active Rust toolchain:** `stable-x86_64-pc-windows-msvc` (the GNU toolchain is a
dead end — the `ort` ONNX crate has no prebuilt GNU binaries).

**Persisted user env vars** (so future builds work in any terminal):
- `LIBCLANG_PATH` = `C:\Users\smart\libclang-bin`
- CMake `bin` dir appended to user `PATH`
- Node, Cargo, MinGW dirs already on user `PATH`

---

## How to Run

```powershell
# From the project root:
.\run.ps1            # dev (hot-reload window)
.\run.ps1 -Build     # production bundle (npm run tauri build)
```

`run.ps1` sets PATH + `LIBCLANG_PATH`, ensures the MSVC toolchain is the rustup
default, and runs the Tauri command. Equivalent manual invocation:

```powershell
$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:LOCALAPPDATA\nodejs\node-v20.18.3-win-x64;C:\Users\smart\cmake\cmake-4.3.3-windows-x86_64\bin;$env:PATH"
$env:LIBCLANG_PATH = "C:\Users\smart\libclang-bin"
npm run tauri dev
```

First clean build ≈ 5–10 min (610 crates + native llama.cpp). Incremental rebuilds
are fast.

---

## How Each Blocker Was Resolved (history)

1. **`linker 'link.exe' not found`** — installed VS Build Tools 2022 with the
   "Desktop development with C++" / VCTools workload (requires admin, one-time).
2. **GNU toolchain dead end** (`ort` has no prebuilt GNU `.lib`) — switched back to
   `stable-x86_64-pc-windows-msvc`.
3. **`Unable to find libclang`** (bindgen, for `llama-cpp-sys-2`) — extracted
   `libclang.dll` from the PyPI `libclang` 18.1.1 win_amd64 wheel (a plain zip, no
   admin) into `C:\Users\smart\libclang-bin`, set `LIBCLANG_PATH`.
4. **`is cmake not installed?`** (`llama-cpp-sys-2` builds native llama.cpp) —
   installed portable CMake 4.3.3 zip, added its `bin` to PATH. CMake uses the
   "Visual Studio 17 2022" generator, detecting the BuildTools instance via vswhere.

---

## Notes

- **Local LLM = Ollama** (replaces the old `local-llm`/`llama-cpp-2` GGUF path).
  Requires the Ollama server running (`run.ps1` auto-starts it) and the model
  pulled: `ollama pull qwen3.5:9b`. Configure model/URL in the app's Settings →
  "Local (Ollama)". Cloud (Gemini) mode is unchanged and needs no Ollama.
- **cmake / libclang:** historically required to build `llama-cpp-2`; **no longer
  needed** now that local generation uses Ollama. The CMake 4.3.3 and libclang
  18.1.1 installs (+ `LIBCLANG_PATH`) remain but are unused by the build.
- **Node version warning:** Vite 7 wants Node ≥ 20.19 but 20.18.3 is installed. It
  runs fine despite the warning; bump Node later if desired.
- **ONNX model:** embedded at `src-tauri/resources/model/model.onnx` (~90 MB) —
  present.
- **MinGW** is installed but unused for the MSVC target; can stay or be removed.

## Quick Verification

```powershell
where.exe link            # MSVC linker
cmake --version           # 4.3.3
Test-Path $env:LIBCLANG_PATH\libclang.dll
rustup show active-toolchain   # ...-msvc
```
