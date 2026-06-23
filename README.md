# Resume & Cover Letter Manager

A lightning-fast, **fully local** desktop application for managing a master repository of
resume experiences, organizing them into targeted professional *archetypes*, and using
Retrieval-Augmented Generation (RAG) to draft highly tailored cover letters from a pasted
job description. It ships with a built-in LaTeX engine for real-time resume compilation and
PDF generation — no TeX Live install required.

Everything — your data, the embedding model, and (optionally) the language model — runs on
your own machine. The only network call is an optional cloud LLM API, which you can turn off
entirely.

---

## Table of Contents

- [Why This Exists](#why-this-exists)
- [Feature Overview](#feature-overview)
- [System Architecture](#system-architecture)
- [How It Works](#how-it-works)
  - [The Data Model](#the-data-model)
  - [The RAG Pipeline](#the-rag-pipeline)
  - [Cover Letter Generation](#cover-letter-generation)
  - [The LaTeX Engine](#the-latex-engine)
  - [PDF Onboarding](#pdf-onboarding)
- [Tech Stack](#tech-stack)
- [Project Layout](#project-layout)
- [Getting Started](#getting-started)
- [Current Status](#current-status)
- [Next Steps & Possible Improvements](#next-steps--possible-improvements)

---

## Why This Exists

Tailoring a resume and cover letter to every application is slow and repetitive. Most people
keep one bloated "master resume" and manually copy-paste the relevant pieces for each job.

This app treats your career as a **structured database of reusable experiences and bullet
points** rather than a single document. You curate the master record once, group the pieces
into role-specific *archetypes* (e.g. "General SWE", "Robotics/Embedded", "ML Engineer"), and
then let semantic search surface the most relevant bullets for any given job description. The
result feeds both an AI-drafted cover letter and a freshly compiled, ATS-friendly PDF resume.

A strict **zero-hallucination policy** ensures the language model may only use experiences that
actually exist in your database — it can never invent skills or employment history.

---

## Feature Overview

| Area | What it does |
|---|---|
| **Master Database CRUD** | Add, edit, and delete experiences (jobs, projects, hackathons, education) and their individual bullet points. |
| **Biographical Info** | Central store for name, email, phone, location, and links (LinkedIn / GitHub / website). |
| **Skills** | Categorized skills (e.g. Languages, Frameworks) taggable per archetype. |
| **Archetype Tagging** | Group bullet points, experiences, and skills into reusable professional profiles. |
| **Semantic Retrieval (RAG)** | Convert a pasted job description into a vector and find the most relevant bullets within a chosen archetype via local KNN search. |
| **Cover Letter Generation** | Build a strict prompt from retrieved bullets and synthesize a tailored letter using a local GGUF model **or** a cloud API. |
| **In-App LaTeX Editor & Compiler** | Split-pane Monaco editor with live PDF preview, compiled by the bundled Tectonic engine. |
| **Template Management** | Ships with a professional ATS-friendly template; injects selected bullets into category-grouped sections. |
| **Resume Layout Controls** | Drag-and-drop section ordering, a 1/2/3-page length selector with automatic spacing calibration, and a minimum-font-size guard. |
| **PDF Onboarding** | Import an existing resume PDF and auto-populate the database via text extraction + LLM structuring. |

---

## System Architecture

The app is a **Tauri v2** desktop application: a Rust backend exposes typed commands over
Tauri's IPC bridge, and a React/TypeScript frontend calls them via `invoke()`. All heavy
lifting — database access, embedding inference, LLM calls, and LaTeX compilation — happens in
the Rust process; the UI stays responsive and never blocks on it.

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Frontend  (React + TypeScript + Vite)            │
│                                                                     │
│   Experiences │ Archetypes │ Generate │ LaTeX │ Bio │ Settings       │
│        │            │           │         │       │       │          │
│        └────────────┴─────── invoke() ────┴───────┴───────┘          │
└────────────────────────────────│────────────────────────────────────┘
                                  │  Tauri IPC (typed commands)
┌────────────────────────────────▼────────────────────────────────────┐
│                          Backend  (Rust)                            │
│                                                                     │
│   db/      → SQLite + sqlite-vec   (experiences, bullets, archetypes)│
│   rag/     → ort + all-MiniLM-L6-v2 ONNX  → 384-d embeddings + KNN   │
│   llm/     → GenerationProvider: Local GGUF (llama-cpp) | Cloud API  │
│   latex/   → Tectonic binary (child process) → PDF bytes            │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                  ┌───────────────┴────────────────┐
            Local SQLite DB                 Bundled ONNX model
         (app data directory)            + downloaded Tectonic engine
```

---

## How It Works

### The Data Model

A local SQLite database (stored in the OS app-data directory, with WAL mode and foreign keys
enabled) holds the entire master record. The schema is created and migrated automatically on
first launch:

- **`experiences`** — title, organization, dates, and a free-form `category` (e.g. Education,
  Professional Experience, Projects).
- **`bullet_points`** — individual achievements, each linked to an experience, with a
  `sort_order`.
- **`archetypes`** — named professional profiles.
- **`archetype_bullets` / `archetype_experiences` / `archetype_skills`** — many-to-many join
  tables that tag content into archetypes.
- **`skills`** — categorized skills.
- **`bullet_embeddings`** — a `sqlite-vec` virtual table (`vec0`) storing one 384-dimensional
  `FLOAT[384]` vector per bullet for semantic search.
- **`bio`** — a single-row table of contact/identity info.
- **`app_settings`** — key/value store (e.g. persisted LLM settings).

> The schema also self-heals: a migration detects and removes an obsolete `CHECK` constraint
> on `experiences.category` so the frontend can use custom categories.

### The RAG Pipeline

Semantic search runs **entirely locally** — no embedding API:

1. The `all-MiniLM-L6-v2` ONNX model and its tokenizer are bundled in
   `src-tauri/resources/model/` and loaded once at startup into Tauri state via the
   [`ort`](https://crates.io/crates/ort) (ONNX Runtime) crate.
2. When a bullet is created or edited, its text is tokenized, run through the model,
   mean-pooled, and L2-normalized into a **384-d vector**, then written to
   `bullet_embeddings` (`INSERT OR REPLACE`).
3. At query time, the job description is embedded the same way, and a KNN query
   (`embedding MATCH ? AND k = ?`) finds the nearest bullets — **filtered to a chosen
   archetype** — ranked by distance.

### Cover Letter Generation

Generation is **dual-mode**, abstracted behind a `GenerationProvider` concept and selected
from persisted settings:

- **Cloud mode (default)** — sends the prompt to the Google **Gemini** REST API using a
  user-supplied API key (`gemini-2.5-flash` by default).
- **Local mode** — loads a user-specified **GGUF** model (Phi-3 recommended) via
  [`llama-cpp-2`](https://crates.io/crates/llama-cpp-2) and runs inference fully offline.
  Compiled behind the `local-llm` Cargo feature.

The prompt enforces the **zero-hallucination policy**: a strict system instruction tells the
model it may *only* use the retrieved bullets and must explicitly call out any required
experience it doesn't have, rather than inventing it.

### The LaTeX Engine

Rather than requiring a multi-gigabyte TeX Live install, the app uses **Tectonic**:

1. On first use, the backend downloads and caches the standalone `tectonic` binary
   (`latex/download.rs`).
2. To compile, the backend writes the `.tex` source to a unique OS temp directory, invokes
   the binary as a child process (`tectonic resume.tex --outdir <tmp>`), and reads back the
   resulting PDF bytes (`latex/mod.rs`). Tectonic fetches any needed LaTeX packages on first
   compile and caches them thereafter.

The default template (`latex/template.rs`) is a single-page, ATS-friendly `article`-class
layout with a maroon accent (`\definecolor{maroon}{RGB}{128,0,0}`). Selected bullets are
**injected** into category-grouped `\section*{}` blocks, and layout commands such as
`\titlespacing` / `\parskip` are calibrated to fit a target page count — with an error
returned if doing so would force a sub-9pt font.

### PDF Onboarding

To bootstrap the database from an existing resume, the onboarding flow uses `pdf-extract` to
pull plain text from an uploaded PDF, then sends it to the cloud LLM to structure it into the
nested experience/bullet hierarchy, which is written back into SQLite.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **App framework** | Tauri v2 |
| **Frontend** | React 19, TypeScript, Vite 7, Tailwind CSS, React Router, `lucide-react` |
| **Editor / Viewer** | `@monaco-editor/react`, `react-pdf`, `@hello-pangea/dnd` |
| **Backend** | Rust |
| **Database** | SQLite via `rusqlite` (bundled) + `sqlite-vec` |
| **Embeddings** | `ort` (ONNX Runtime) + `all-MiniLM-L6-v2`, `tokenizers` |
| **Local LLM** | `llama-cpp-2` (optional `local-llm` feature) |
| **Cloud LLM** | Google Gemini via `reqwest` |
| **LaTeX** | Tectonic (downloaded standalone binary) |
| **PDF parsing** | `pdf-extract` |

---

## Project Layout

```
resume-builder/
├── src/                        # React frontend
│   ├── components/Layout.tsx   # Sidebar nav + content shell
│   ├── pages/
│   │   ├── ExperiencesPage.tsx # Experience + bullet CRUD
│   │   ├── ArchetypesPage.tsx  # Tag/untag content into archetypes
│   │   ├── GeneratePage.tsx    # JD → RAG → cover letter
│   │   ├── LatexPage.tsx       # Monaco editor + PDF preview + layout controls
│   │   ├── BioPage.tsx         # Contact/identity info
│   │   ├── OnboardingPage.tsx  # Import resume PDF
│   │   └── SettingsPage.tsx    # LLM mode / model path / API key
│   └── lib/tauri.ts            # Typed invoke() wrappers
│
├── src-tauri/                  # Rust backend
│   ├── src/
│   │   ├── db/                 # Schema, migrations, CRUD commands
│   │   ├── rag/                # Embedding model + semantic search
│   │   ├── llm/                # Dual-mode generation + prompt + settings
│   │   ├── latex/              # Tectonic download, compile, template injection
│   │   └── lib.rs             # Tauri setup + command registration
│   ├── resources/model/        # Bundled ONNX model + tokenizer
│   └── Cargo.toml
│
├── prd.md                      # Product requirements
├── implementation_plan_exported.md
├── handoff.md                  # Dev-environment setup notes
└── memory.md                   # Running log of decisions/phases
```

---

## Getting Started

### Prerequisites

- **Node.js** ≥ 20.19 (the project has been run on 20.18.3 with a Vite warning)
- **Rust** (stable) with the **MSVC** toolchain on Windows
- **Visual Studio Build Tools** with the "Desktop development with C++" workload — required
  because the `ort` and `llama-cpp-2` crates need the MSVC linker (`link.exe`) and a C++
  compiler. The GNU toolchain is **not** supported (`ort` ships only MSVC prebuilts).
- A **CMake** install is needed if building with the local LLM feature.

> See [`handoff.md`](handoff.md) for a detailed, no-admin-friendly walkthrough of installing
> the toolchain on Windows.

### Run in development

```powershell
npm install
npm run tauri dev
```

The first Rust build compiles ~600 crates and takes several minutes; subsequent builds are
incremental. The first LaTeX compile downloads the Tectonic engine and package cache.

### Build a release bundle

```powershell
npm run tauri build
```

### Cloud-only build (skip the local LLM)

The `local-llm` feature is enabled by default. To build without it (avoids the C++/CMake
dependency and uses cloud generation only), disable default features for the Tauri crate when
invoking the build.

### Configure generation

Open **Settings** in the app to choose **Cloud** (paste a Gemini API key) or **Local** (point
to a GGUF model file). Settings are persisted in the local database.

---

## Current Status

All core PRD features are implemented in code and the Rust layer has unit/integration tests
(DB CRUD, cascade deletes, archetype tagging, and `sqlite-vec` KNN). Implemented phases per
[`memory.md`](memory.md):

- [x] Scaffolding (Tauri v2 + React/TS + Vite)
- [x] Database layer (SQLite + `sqlite-vec`)
- [x] RAG pipeline (ONNX embeddings + KNN)
- [x] Dual-mode LLM (cloud Gemini verified; local GGUF behind a feature flag)
- [x] LaTeX engine (Tectonic download + compile + injection)
- [x] Frontend (all pages wired to backend)
- [x] Layout enhancements (category sections, drag-and-drop ordering, page-length selector)
- [x] PDF onboarding

**Known caveats:**

- Building on Windows requires the MSVC Build Tools (see above) — this is the main setup
  blocker, not missing functionality.
- The **cloud (Gemini)** generation path is the verified default; the **local GGUF** path is
  implemented but less exercised and depends on a user-provided model + CMake-built native
  code.

---

## Next Steps & Possible Improvements

**Robustness & polish**
- End-to-end integration testing of the full flow (add experience → generate letter → compile
  PDF → export), plus loading indicators for embedding, LLM inference, and the Tectonic
  first-run download.
- Graceful error handling for missing GGUF models, invalid API keys, empty database, and
  LaTeX compilation failures (surfacing Tectonic's error output in the UI).
- Verify and harden the local LLM path; consider bundling or guided-downloading a default
  Phi-3 GGUF so local mode works out of the box.

**Features**
- Cover-letter editing/versioning and the ability to save generated letters back into the
  database.
- Multiple resume templates and a template gallery; richer custom-template import via string
  replacement.
- Streaming LLM output token-by-token for faster perceived generation.
- Export/import of the entire database for backup and cross-device sync.
- Per-bullet relevance scores and an interactive "include/exclude" step before generation, so
  the user can curate retrieved bullets.
- Additional cloud providers (e.g. Anthropic Claude, OpenAI) behind the existing provider
  abstraction.

**Performance & quality**
- Cache embeddings and avoid re-embedding unchanged bullets on bulk operations.
- Move long-running backend work onto dedicated async tasks with progress events to keep the
  UI perfectly responsive under load.
- Re-rank RAG results (e.g. with a cross-encoder) for higher-precision retrieval.

**Distribution**
- Code signing and auto-update for the packaged desktop app.
- Cross-platform builds (macOS / Linux) and CI to produce release artifacts.
```