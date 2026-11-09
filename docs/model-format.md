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
| `scenarios` | Exactly two: `aligned` and `padded`, each with a `params` map of positive integers and optional `label`. Aligned output dimensions divide by tile size; padded has at least one tail output tile. |

The viewer supports Chinese and English. Keep the existing English `title` and `label` fields; optional nonempty `title_zh`, `label_zh`, and `mapping.note_zh` provide Chinese text for the model title, scenarios, program steps, address samples, and unknown-mapping explanation. Missing Chinese text falls back to the English field. Source code, axis names, formulas, and evidence keys are never translated.

Expressions accept integer constants, named parameters, `wid_x/y/z`, `tile_<axis>`, `phase`, `+`, `-`, `*`, `//`, `%`, `ceildiv(a,b)`, `min(...)`, and `max(...)`. They reject arbitrary Python, attributes, indexing, powers, and unknown names. Address `sample` variables are evaluated in their listed order; a later variable may refer to an earlier one. `phase` is the zero-based loop iteration.

The validator expands every worker and requires **exactly one worker for every output tile**. It rejects duplicate or missing tiles, out-of-range coordinates, invalid source evidence, unsupported formulas, and cyclic dependencies. Limits keep the browser demo small: at most 256 tiles and 96 expanded stages per scenario.

For a 2D output, the first tile axis is displayed as rows and the second as columns. For a 1D output, tiles form one row. A tile's padded cells are determined by output bounds. A repeated K loop can separately show a masked reduction slice through `loop_valid`.

## Agent extraction boundary

Read both the kernel and its launcher or host tiling setup. If shape or tile sizes are absent, choose small **example** dimensions that preserve the source formulas, then choose a second set with a tail. Do not claim that those sample shapes were measured or provided by the user. Every mapping, stage, and address expression must cite a matching source line. Literal evidence makes the model reviewable; it does not establish full semantic equivalence. If the source uses dynamic branching, opaque pointer arithmetic, or cross-tile synchronization that cannot be reduced to these rules, return an `unknown` model with an explanation instead of inventing an animation.
