# Tile model format (v1)

Use [cuda_tiled_gemm.json](../examples/cuda_tiled_gemm.json) as a 2D example and [triton_ascend_add.json](../examples/triton_ascend_add.json) as a 1D example.

## Required fields

| Field | Meaning |
| --- | --- |
| `schema_version` | Integer `1`. |
| `title`, `backend` | Display title; backend is `cuda`, `triton-cuda`, or `triton-ascend`. |
| `source.path`, `source.kernel` | Local source path (relative to the JSON file or absolute) and kernel name. |
| `source.refs` | Named evidence references with one-based `line` and an exact `contains` substring from that line. |
| `mapping.status` | `source-backed` for inspectable rules, or `unknown` with a `note`. Unknown models do not play. |
| `mapping.worker_grid` | Integer formulas for `x`, optional `y` and `z` (default `1`). `wid_x`, `wid_y`, `wid_z` enumerate these workers. |
| `mapping.grid_source_ref` | Evidence key for launch grid construction. |
| `mapping.tile_axes` | One or two output axes. Each has `name`, `extent`, `size`, `index`, and `source_ref`. `index` maps a worker to a logical tile coordinate. |
| `mapping.program` | Ordered groups of steps. Each group has a nonempty `steps` list with `id`, `label`, `source_ref`. A repeated group adds `repeat`, `loop_axis`, `loop_tile`, `loop_source_ref`. |
| `mapping.addresses` | Optional sample offset formulas. Each has `label`, `expr`, `sample` variable formulas, and `source_ref`. These are element offsets, not byte addresses. |
| `mapping.dependencies` | Optional list of `{ "tile": [coordinates], "on": [[predecessor coordinates]] }`. Coordinates follow `tile_axes` order. |
| `mapping.gemm_view` | Optional A × B → C slice diagram for a 2D GEMM. Declares output `m_axis`/`n_axis`, integer `k_extent`/`k_size`, source refs for A/B/MMA/C, and `step_roles` mapping stage IDs to highlighted matrices. |
| `scenarios` | Exactly two: `aligned` and `padded`, each with a `params` map of positive integers and optional `label`. Aligned output dimensions divide by tile size; padded has at least one tail output tile. |

The viewer supports Chinese and English. Keep the existing English `title` and `label` fields; optional nonempty `title_zh`, `label_zh`, and `mapping.note_zh` provide Chinese text for the model title, scenarios, program steps, address samples, and unknown-mapping explanation. Missing Chinese text falls back to the English field. Source code, axis names, formulas, and evidence keys are never translated.

Expressions accept integer constants, named parameters, `wid_x/y/z`, `tile_<axis>`, `phase`, `+`, `-`, `*`, `//`, `%`, `ceildiv(a,b)`, `min(...)`, and `max(...)`. They reject arbitrary Python, attributes, indexing, powers, and unknown names. Address `sample` variables are evaluated in their listed order; a later variable may refer to an earlier one. `phase` is the zero-based loop iteration.

The validator expands every worker and requires **exactly one worker for every output tile**. It rejects duplicate or missing tiles, out-of-range coordinates, invalid source evidence, unsupported formulas, and cyclic dependencies. Limits keep the browser demo small: at most 256 tiles and 96 expanded stages per scenario.

For a 2D output, the first tile axis is displayed as rows and the second as columns. For a 1D output, tiles form one row. A tile's padded cells are determined by output bounds. A repeated K loop can separately show a masked reduction slice through `loop_valid`.

The optional `gemm_view` uses the same validated output tile bounds and K-loop phase to draw A[M tile,K slice], B[K slice,N tile], and C[M tile,N tile]. It checks that the K slice count matches the declared loop, and that source refs and highlighted step IDs exist. See [cuda_tensor_core_gemm.json](../examples/cuda_tensor_core_gemm.json). Matrix cells are a logical view; WMMA fragment register ownership is intentionally not inferred, because NVIDIA leaves that mapping unspecified.

## Agent extraction boundary

Read both the kernel and its launcher or host tiling setup. If shape or tile sizes are absent, choose small **example** dimensions that preserve the source formulas, then choose a second set with a tail. Do not claim that those sample shapes were measured or provided by the user. Every mapping, stage, and address expression must cite a matching source line. Literal evidence makes the model reviewable; it does not establish full semantic equivalence. If the source uses dynamic branching, opaque pointer arithmetic, or cross-tile synchronization that cannot be reduced to these rules, return an `unknown` model with an explanation instead of inventing an animation.

## v2: source-backed logical task trace

Start from [ascend_qkv_alltoall.json](../examples/ascend_qkv_alltoall.json). This first v2 implementation supports `trace.kind: "qkv-head-grouped"` for the cited `kernel_qkv_fuse_a2a_head_grouped` branch. It is a bounded, deterministic description for this loop structure, not a parser for arbitrary CUDA or Triton source.

| Field | Meaning |
| --- | --- |
| `schema_version` | Integer `2`. |
| `title`, `backend`, `source`, `mapping.status` | Same evidence and unknown-state rules as v1. |
| `trace.kind` | `qkv-head-grouped`. |
| `trace.formulas` | Exact formula keys from the example: lane, task index, target rank, head group, block, consumer flat index, signal slot/offset, output row, Q/K/V peer offsets, input/output offsets. |
| `trace.formula_refs` | One source evidence key for each formula. |
| `trace.refs` | Source evidence keys for role and wave loops, load, direct/peer store, fence, notify, wait, consume, output store. |
| `scenarios` | `aligned` and `padded` example cases. Each `params` includes sequence/rank/head/core counts, wave and tile sizes, group size, signal stride, and explicit input/output strides. The bundled examples choose contiguous strides. |

The validator checks every expanded task against the known source loop layout, including `flat_idx → (block, target rank, group) → Producer PID`, the inverse Consumer claim, exact output tile coverage, mask counts, Q/K/V peer offsets, signal slot, and notify/wait pairing. It rejects unsupported formulas, missing dimensions, incorrect source lines, and nonmatching offsets. It caps expansion at 2048 tasks and 10000 events per case. Formula syntax is the same safe integer subset as v1.

Each expanded task has a distinct execution-unit PID and logical tile ID. It also carries a route (`local` or `remote`), valid token count, output row, peer offsets, signal slot, and event IDs. Events form a dependency DAG. The browser advances all ready events at each logical level, and displays their causal order rather than device timing. A source-backed status should be used only after checking the source and launcher; `unknown` disables playback when a relation cannot be established.

The bundled QKV example includes a copy of its source and upstream license in `examples/sources/`, so its line refs remain stable. With 4 ranks, 20 heads per rank, 32 AIVs, `S_WAVE=4`, `BLOCK_S=112`, and `BLOCK_D=128`, `S=1344` produces three full waves. `S=1300` has 68 valid tokens in block 11, with 44 masked tokens. These numbers are demonstration inputs.
