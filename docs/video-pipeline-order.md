# Variable pipeline order

## Complete order: fourteen phases

All processing tabs now have reorder handles, including Outpainting, Shot Detection,
References, Colorization, Recomposition, Audio and Upscaling. Overview has arrows
for its workflow tiles; the individual Shots/References/Recomposition phases are
reordered through their tabs. Disabled phases retain their saved positions.

The current UI saves `global.pipeline_order`, a permutation of these fourteen IDs:

```
denoise,deflicker,enhancement,cleanup,stabilize,dust,detail,outpaint,shots,references,colour,recomp,audio,upscale
```

Old `video_order` and `preprocessing_order` values are retained and read without
rewriting a project. The first complete reorder starts from the existing video
order followed by the existing downstream phases. Once `pipeline_order` is saved,
it is authoritative. Save requests are project-bound and refused while busy.

### Actual bindings, not just display order

The plan exposes each enabled phase's video input/output and artifact bindings.
Shots and References produce/consume metadata without replacing the current video.
Colorization consumes the manifest/reference set associated with its exact input.
Outpainting records its original centre and generated canvas; Recomposition binds
those explicitly, along with a matching colour result when available. The next
video transform, Audio or Upscaling receives the actual preceding video result.
The last video—not necessarily Upscaling—is the final output.

Examples supported by the binding contracts:

- Upscale → Outpaint → Shots → References → Colorize → Recompose → Audio.
- Shots → References → Colorize → Recompose → Outpaint → Audio → Upscale.
- Audio after Upscaling, or video transforms after Recomposition.

Every phase is movable, but **not every permutation is executable**. Inserting a
video transform between reference generation and Colorization requires matching
Shots/References after that transform. Recomposition cannot use a canvas/colour
branch from before an intervening transform, or combine unrelated canvases.
Invalid placements remain saved and editable: Overview explains the structural
problem and rendering is blocked without sorting phases back or generating hidden
dependencies. These conservative checks also reject audio inserted between a
paired image branch and its recomposition; move Audio after that recomposition.

### Artifact safety and readiness

Full-order shot manifests use a `_pipeline_<identity>` suffix keyed on the actual
video path/fingerprint and Shot Detection settings. Reordering does not rewrite
an existing authored shot list. Matching old manifests are not copied from another
outpaint model. Actual manifest source paths are checked before reference/colour
processing; Recomposition also checks branch frame counts and cadence. Normal
model-resolution differences are not mistaken for timeline changes.

Finished artifact phases must have matching producer stamps before downstream
execution. Unknown, stale or missing artifacts are not substituted from arbitrary
available files. “Keep shots” cannot override these checks after reordering.
Direct per-shot reference regeneration, reference editing, guide generation,
outpaint-chunk rendering and upscale previews use the same binding guards before
any model/server startup. All commercial/licence gates remain in force.

Upscaling uses an optional shot manifest only when Shot Detection precedes it.
When Upscaling comes first, no later or unrelated legacy shot list is passed to
the backend or shown as its per-shot controls. The Shots tab uses the same scoped
manifest as its planned producer, including when Colorization is disabled.

Ordering itself does not render, download models, delete media or adopt outputs.
Derived input pointers are updated, while processing parameters remain unchanged.
Explicit later rendering still follows each existing backend's regeneration and
cache policy; this change is not a new immutable versioning system for every model
output. Review newly bound shot/reference data before a long whole-pipeline run.

### Verification and limits

105 focused Python tests and 102 JS tests pass, including the previous seven-stage
coverage, full-order parsing/migration, valid/invalid artifact permutations,
planned command inputs, scoped shot-manifest outputs/publication stamps, optional
upstream-only Upscaling shot bindings, project/busy guards and
direct-model-entry rejection. The real headless-browser navigation fixture checks
14 handles, actual tab reordering, legacy-field preservation and no page overflow
at 1440/768/390. `git diff --check` passes.

The new full-model paths are **contract-tested with fixtures/test doubles**, not
validated by an end-to-end GPU remaster. No new model inference, remote API/cloud
job, live-project migration or active-server restart was performed. Earlier CPU
round trips remain separate evidence; no historical restoration-quality conclusion
follows. The broad GUI suite's outstanding errors remain unresolved.

## Earlier seven-stage implementation / compatibility

The seven single-video transforms can now move using Overview arrows or the
existing tab drag/reorder menus:

- Remove Noise (`denoise`)
- Light Deflicker (`deflicker`)
- Detail Enhancement / Deblur (`enhancement`)
- Clean Up (`cleanup`)
- Stabilization (`stabilize`)
- Dust & Scratches (`dust`)
- legacy Detail Enhancement (`detail`)

This is an actual execution/input order, not a tab-only preference. Disabled
stages retain their positions but do not consume or produce a video. Models,
processing defaults, licence guards and Dust's explicit adoption stay unchanged.
Order changes neither start inference nor delete existing media. Changing order
while a worker or the whole-remaster queue is running is refused.

## Saved settings and compatibility

`global.video_order` is a comma-separated permutation of all seven keys, including
disabled stages. Duplicate, missing, unknown or malformed entries are rejected.
The earlier seven-stage UI saved this field; the current UI saves `pipeline_order`
as described above. Without either field, old projects use their existing
three-stage `global.preprocessing_order` followed by Clean Up, Stabilization, Dust
and legacy Detail, exactly as before. Reading a project does not rewrite it.
The old three-stage field is retained, not overwritten; the seven-stage field
takes precedence once explicitly saved. Older ARP versions do not understand
that field and cannot execute the new seven-stage arrangement.

For compatibility, the old three-stage standalone original-source preview fallback
remains only on projects without `video_order`. Explicitly reordered projects
are strict: missing/unverified predecessors block both standalone jobs and queue
jobs. They never substitute the original or an unrelated available intermediate.

## One chain, multiple consumers

The common order fold drives active stage order, command `--source`, expected
outputs, comparison inputs, provenance labels, readiness checks and the terminal
video consumed by Outpainting/the fixed downstream phases. Legacy-named source
helpers act as adapters to the configured consumer's input; they no longer impose
their namesake stage's former position in reordered projects.

Signed readiness is checked in configured predecessor order. Dust additionally
requires its current mask-gated report to be explicitly adopted. Moving Dust
ahead of Stabilization does not require a later stabilization result first.
Changing Dust's actual input invalidates its previous adoption; it does not delete
the old analysis or repair. A move that leaves effective inputs/settings identical
may reuse matching work; moving disabled stages is not itself a reason to render.

Clean Up and legacy Detail require matching source provenance and timing;
Detail's signature is checked against its current parameters, while Clean Up's
identity, format and chunk settings must match. Missing or stale video predecessors
are shown as waiting, and the terminal output cannot silently select an earlier
video stage merely because its file exists. Derived downstream input links are
refreshed/cleared by the existing hydration mechanism; authored media are retained.
Clean Up/Dearchive may legitimately change resolution at its configured model
height. Its downstream stages consume that actual output; readiness does not
pretend it is still the original raster. Legacy Detail must retain geometry.

## Coverage and remaining scope

Projects with only the older `video_order` field retain Outpainting, Shot Detection,
References, Colorization, Recomposition, Audio and Upscaling in their previous
downstream positions. The complete-order field above enables the second block's
explicit artifact/timeline contracts and dependency validation.
This feature does not assert that every processing order produces useful quality
or that existing algorithm format/geometry constraints disappear.

Validation covers all **5,040 permutations × 128 enabled subsets** in the pure
chain fold, command/input agreement across rotated/reversed seven-stage orders,
legacy migration, busy-state rejection, strict predecessor checks and a real CPU
Detail → Dust round trip. Reversing that pair invalidates adoption while preserving
the previous repaired file's hash. A separate headless-browser fixture exercises
seven reorder handles, moving Dust to first, legacy-field preservation and the
busy guard at 1440/768/390, without using an active film project or running models.

No GPU run, historical quality evaluation or full live-project reorder migration
was performed for this block. GitNexus status timed out; call-site verification
was performed directly in source and tests rather than relying on its stale index.
The focused regression suites passed **89 Python tests** and **102 JS tests**;
the isolated real-browser reorder check and `git diff --check` passed as well.
The broader GUI suite's outstanding errors were not resolved by this block.
