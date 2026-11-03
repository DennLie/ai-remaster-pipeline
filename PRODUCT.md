# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Archive film restorers, semi-professional to professional, working on public-domain or licensed footage. They run ARP on a single Windows workstation with an NVIDIA GPU (24 GB recommended, 16 GB floor) and treat restoration as serious craft rather than a novelty. Committed enthusiasts are welcome and use the same defaults, which favour quality over convenience.

Their job: take an old film and produce a faithful, watchable master — cleaned, stabilised, extended to a modern aspect ratio, colourised, scored, and upscaled — while keeping control of every AI decision that could distort the source. A feature-length, multi-stage restoration can run for days to months, so work happens in repeated review loops per shot rather than one unattended render.

## Product Purpose

ARP is a local GUI that orchestrates ComfyUI models through a complete film-remaster pipeline on one page. The user browses to a video or numbered image sequence, ticks the stages they want, runs the whole pipeline, then opens each stage's tab to inspect output and rerun only what needs fixing.

Success is a faithful, reviewable restoration: the result respects the source, and the user can see and steer every AI decision that shaped it. Speed matters, but never at the expense of that reviewability.

## Positioning

Three mechanisms a neighbouring video tool could not truthfully copy without rebuilding its model around them:

- **The original pixels win.** AI output never replaces source footage that already exists; recomposition restores the untouched source in the centre and blends it into generated edges.
- **Stills decide colour; video models spread it.** Colour is chosen per shot on stills that are quick to review and redo, and the video colouriser only has to follow those references — including showing Qwen 2511 a shot's existing references so long shots stay consistent.
- **Shots are the unit of work.** References, strengths, fades, and upscale strength live on the shot list, so a fix touches only the shots it changes and reruns reuse everything that still matches.

## Operating Context

- Windows + NVIDIA is the supported path; installation builds a project-local `.venv`, local FFmpeg, and a managed ComfyUI checkout, and launch opens the GUI in the browser from a local server.
- Users accept model access terms on Hugging Face for gated models before first use.
- Models download lazily per stage; the full set is roughly 85-100 GB on top of the runtime, and projects can consume hundreds of GB of intermediates.
- Stages run in sequence with a review gate between them: shot-boundary fixes, reference selection and regeneration, per-shot strength tuning, then final output in `output/reassembled/`.
- Any stage can be delegated to a RunPod cloud worker instead of the local GPU; otherwise the machine is the whole system.
- Backend wrappers remain runnable directly from `wrappers/` for users who prefer the command line, and the GUI shows each stage's command before running it.
- Model licences constrain use: ProPainter and Qwen-Image-2.1 are non-commercial, MiniMax H3 is region-restricted, and ARP itself grants no rights to films, weights, LoRAs, or third-party nodes.

## Capabilities and Constraints

- Nine stages: Clean Up (DeVignette, scratch repair, archive restoration), Stabilization, Outpainting, Shot Detection, Reference Generation, Colorization, Recomposition, Audio Track, Upscaling.
- Multiple model choices per stage, with GGUF quantisation, offloaded text encoders, chunked rendering, and tiling to fit a 24 GB card.
- Every output records a signature of its inputs and settings; reruns reuse matching work down to the chunk.
- Intermediates are 10-bit VP9 or FFV1 masters; joins and trims are stream copies and every join is frame-count checked.
- Alpha software. Python 3.13, ComfyUI backend, single local user, no accounts or hosted service.
- Not committed: support beyond Windows + NVIDIA, a business model beyond Apache-2.0 licensing and an optional donation link, and any hosted or multi-user mode.

## Brand Commitments

- Name: **ARP — AI Remaster Pipeline**, with existing logo and app-icon assets under `assets/branding/`.
- Voice in existing docs: plain-spoken, technically candid, quick to state limits (alpha status, hardware floors, licence restrictions, time estimates in days to months). No marketing superlatives and no claims the pipeline cannot demonstrate.
- Apache-2.0 licensed; "Buy Me a Coffee" is the only support channel presented.

## Evidence on Hand

- Real GUI screenshots for Overview, Shot Detection, Reference Generation, Colorization, Recomposition, and Output in `assets/screenshots/walkthrough/`.
- Substantive technical documentation in `docs/` — user guide, per-stage research, benchmark comparisons, licence audit, and model-source notes.
- Working workflows under `workflows/` and CI jobs under `.github/workflows/` covering denoise, deflicker, and commercial-model guard checks.
- Absent, and not to be invented: testimonials, customer names, benchmarks presented as marketing proof, pricing, or delivery guarantees.

## Product Principles

1. **Faithful and reviewable beats convenient.** Every AI decision that could alter the film stays visible and adjustable.
2. **The source is the reference.** Existing footage is never silently replaced; generated content fills only where the user asked for it.
3. **Shots are the unit of work.** Review, tuning, and reruns stay local to the shots they change.
4. **Everything resumes.** No needless generations; matching work is reused down to the chunk.
5. **One consumer GPU is enough.** Capability is delivered within 24 GB, not deferred to datacentre hardware.
