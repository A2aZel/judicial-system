# Docker for Mac Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the existing structured/dense kNN application locally and ship a tested CPU Docker image for Apple Silicon and Intel Macs, including an 8 GB Mac configuration.

**Architecture:** Keep application behavior and UI unchanged. Build native Linux arm64/amd64 images from architecture-specific hashed CPU locks, smoke-test each running image, then combine their tested registry digests into a multi-platform image. Keep private processed data and downloaded model weights outside the image.

**Tech Stack:** Python 3.12, FastAPI, CPU PyTorch, Docker Compose, native GitHub Actions runners, GHCR.

**Spec:** User-approved session requirements; packaging decision in `docs/adr/0002-docker-mac.md`.

## Global Constraints

- Work only on `feat/docker-mac-knn` in the isolated worktree based on `origin/main` (`10824ce`); do not include dirty source-workspace corpus changes.
- Preserve existing structured + dense kNN, API/UI and pinned encoder revisions. No Qwen downloads or generation; appeal risk disabled.
- Publish `ghcr.io/a2azel/judicial-system:mac8gb` and commit SHA tags only from the dedicated workflow; never overwrite `latest`.
- Trigger the dedicated workflow on pushes to `main` and `feat/docker-mac-knn`, plus manual dispatch. Use `GITHUB_TOKEN` with `packages: write`.
- Do not bundle private corpus, processed data, model caches, credentials or weights. Include only the explicit synthetic fixture and required runtime code.
- No commit/push by implementation workers; lead handles authorized delivery after verification.

## Review Focus

- arm64 must select a compatible CPU wheel lock; verify hashed dependency installation in the native arm64 build.
- A process that starts with an empty corpus is insufficient; smoke requires fixture records and finite, ranked analysis results.
- One failed architecture must prevent release tags; merge only the two successfully tested digests.
- Real processed data includes generation files; document copying the complete processed directory and verify readonly mounts.
- An 8 GB Mac has a limited Docker VM budget; cap the service at 4 GB / 2 CPUs, dense batch size 1, and describe unmeasured memory/latency limits accurately.

### Task 1: Portable CPU runtime (runtime worker)

**Files:** `Dockerfile`, `.dockerignore`, `requirements-arm64-lock.txt`, `compose.mac.yaml`, `.env.mac.example`, `tools/docker_smoke.py`.

**Interfaces:** Image starts on port 8000 with an included synthetic CSV. Standalone Mac Compose mounts `./data` readonly at `/data`, uses `/data/processed` for retrieval/index files, and stores Hugging Face cache outside the image. Smoke interface: `python tools/docker_smoke.py --base-url http://127.0.0.1:8000` (stdlib only).

- [ ] Resolve an arm64 CPU-only hashed lock and use the correct lock for each target architecture; pin a multi-platform Python base digest.
- [ ] Restrict Docker context to explicit runtime files; add the Mac profile with the agreed resource/default settings.
- [ ] Implement HTTP smoke for health, model metadata, UI and a real analysis against the synthetic fixture; no model downloads needed.
- [ ] Verify Compose interpolation, build a local image and run smoke against the live container; inspect image/context for forbidden data and dependencies.

### Task 2: Tested multi-platform release and instructions (CI/docs worker)

**Files:** `.github/workflows/docker-mac.yml`, existing `.github/workflows/ci.yml` if needed, `README.md`, `docs/adr/0002-docker-mac.md`, this plan.

**Interfaces:** Consume the runtime worker's image and smoke CLI. Use `ubuntu-24.04` for amd64 and `ubuntu-24.04-arm` for arm64. Publish only tested per-architecture digests, then create the manifest and verify both platforms.

- [x] Implement native build/load, container HTTP smoke, then push that exact tested image and record its digest.
- [x] Pass only validated digest files between jobs; merge after both jobs pass into `mac8gb` plus commit SHA tag and inspect the result.
- [x] Replace the obsolete amd64-only README claim and document source/published startup, logs, stop, complete processed-data transfer, cache and Mac resource limits.
- [x] Validate both workflows with actionlint 1.7.12 and both Compose files with `config --quiet`; review the final diff. Native GitHub build/publication remains Task 3.

### Task 3: End-to-end verification and handoff (lead)

- [ ] Start the existing local application first; verify real structured/dense kNN and UI, and record measured results without exposing case text.
- [ ] Run the relevant Python/UI regression suite, Docker build/container smoke, context/privacy inspection and independent review.
- [ ] Report the measured local result separately from unexecuted Mac/native CI checks. Commit/push only the approved feature branch, verify GitHub checks/publication, and hand off the concrete branch/image instructions.
