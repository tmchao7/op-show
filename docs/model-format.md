# Tile model format (v1 and v2)

A model is a JSON file plus a local operator source. v1 describes output tiles, the worker-to-tile mapping, per-tile program stages, and sample address offsets. v2 describes a bounded logical task trace for one fixed loop structure. Use [cuda_tiled_gemm.json](../examples/cuda_tiled_gemm.json) as a 2D v1 example, [triton_ascend_add.json](../examples/triton_ascend_add.json) as a 1D v1 example, and [ascend_qkv_alltoall.json](../examples/ascend_qkv_alltoall.json) as a v2 example.

## Common fields

| Field | Meaning |
| --- | --- |
| `schema_version` | Integer `1` (tile model) or `2` (task trace). |
| `title`, `title_zh` | Display title. `title_zh` is optional; missing Chinese text falls back to English. |
| `backend` | `cuda`, `triton-cuda`, or `triton-ascend`. |
| `source.path`, `source.kernel` | Local source path (relative to the JSON file, or absolute) and kernel name. The validator checks that the name occurs in the source text. |
| `source.refs` | Named evidence references. Each has a one-based `line` and an exact `contains` substring of that line. Any field that cites evidence takes one of these keys. |
| `mapping.status` | `source-backed` for inspectable rules, or `unknown`. An unknown model requires `mapping.note` and disables playback. |
| `mapping.note`, `mapping.note_zh` | Optional explanation of the declared mapping. `note` is required for unknown models; `note_zh` is its Chinese text and is accepted for both source-backed and unknown mappings. |

Source code, axis names, formulas, and evidence keys are never translated.

## v1: output tiles

| Field | Meaning |
| --- | --- |
| `mapping.worker_grid` | Integer formulas for `x`, optional `y` and `z` (default `1`). `wid_x`, `wid_y`, `wid_z` enumerate these workers. |
| `mapping.grid_source_ref` | Evidence key for launch grid construction. |
| `mapping.worker_labels` | Optional display labels for the worker axes, for example `{"x": "blockIdx.x", "y": "blockIdx.y"}`. The viewer shows them next to worker IDs; they are display text, not source evidence. |
| `mapping.claim_loop` | Optional `{ "count": integer formula, "source_refs": [evidence keys] }`. Each worker executes `claim_i = 0..count-1` in source loop order. Omit for one output task per worker. The cited lines must cover the loop structure; the validator expands claims and chains the same worker's output tasks. |
| `mapping.tile_axes` | One or two output axes. Each has `name`, `extent`, `size`, optional `size_ref`, `index`, and `source_ref`. `index` maps a `(worker, claim_i)` pair to a logical tile coordinate; `extent` and `size` are integer formulas. |
| `mapping.parallelism` | Optional source-cited explanation: `claim_unit`, optional `claim_unit_zh`, `summary`, optional `summary_zh`, and nonempty `source_refs`. Optional `task_group` has `label`, optional `label_zh`, `count`, `index`, `source_ref`, and optional `member_axis`; it assigns each output task to a shared input group such as a KV head. `member_axis` names the output axis whose members should be listed under each group (GQA uses `H`). The validator checks the group index and axis. |
| `mapping.program` | Ordered groups of steps. Each group has a nonempty `steps` list with `id`, `label`, and `source_ref`; `label_zh` is optional. A repeated group adds `repeat`, `loop_axis`, `loop_tile`, and `loop_source_ref`. |
| `mapping.result_progress` | Optional `{ "loop_source_ref": evidence key, "contribution_step": step ID }` for a source-backed reduction loop. The loop ref must identify exactly one group, and the named step must occur exactly once in each iteration. The viewer shades the result tile by valid reduction positions processed after that step, independently of output completion. |
| `mapping.addresses` | Optional sample offset formulas. Each has `label`, `expr`, `sample` variable formulas, and `source_ref`. These are element offsets, not byte addresses. |
| `mapping.dependencies` | Optional list of `{ "tile": [coordinates], "on": [[predecessor coordinates]] }`. Coordinates follow `tile_axes` order. |
| `mapping.gemm_view` | Optional A × B → C slice diagram for a 2D GEMM. Declares output `m_axis`/`n_axis`, integer `k_extent`/`k_size`, source refs for A/B/MMA/C, and `step_roles` mapping stage IDs to highlighted matrices. |
| `mapping.row_k_view` | Optional 1D output / K-slice diagram for a row reduction or GEMV. Declares `kind` (`reduce` or `gemv`), `output_axis`, `k_extent`/`k_size`, source refs, and stage-to-input/output `step_roles`. |
| `mapping.attention_view` | Optional Q/K/V/S/O slice diagram for a 2D attention output. Declares `q_axis`/`d_axis`, key length `n_extent`/`n_size`, source refs for Q/K/V/score/softmax/O, and `step_roles`. |
| `scenarios` | Exactly two: `aligned` and `padded`, each with a `params` map of positive integers and optional `label`/`label_zh`. Aligned output dimensions divide evenly by tile size; padded has at least one tail output tile. |

### Repeated groups and loops

A repeated group must satisfy `repeat == ceildiv(params[loop_axis], params[loop_tile])`; `loop_axis` and `loop_tile` must name parameters, and `loop_source_ref` must be an evidence key. The validator rejects a repeat count that does not match the declared loop extent, so the stage list cannot silently omit or add iterations. When `repeat` is absent it defaults to `1`. A group with `repeat == 1` is emitted as one iteration, and its expanded steps carry `null` for `iteration`, `loop_axis`, `loop_tile`, and `loop_valid`; if loop fields are declared anyway, they must still be consistent with the repeat count and cite `loop_source_ref`.

For a repeated group, each iteration derives `loop_valid = max(0, min(params[loop_tile], params[loop_axis] - iteration * params[loop_tile]))`. That is how the last K slice can be shown as a masked reduction slice. `phase` is the zero-based loop iteration.

When `result_progress` is declared, the validator derives checkpoints from the original loop group, including the single-iteration case whose expanded steps have `null` loop fields. Each checkpoint records its expanded `step_index` and `processed = min(axis extent, (iteration + 1) × loop tile)`. For a tile, the browser advances its shade only after `tick > tile.start + step_index`. The seven-level warm color scale (including zero) and the exact `processed/extent` text represent logical reduction coverage, not accumulator values or a global-memory write. A tile can reach full reduction coverage before its epilogue or store finishes. For row reduction and GEMV this can still be a lane-local sum; for Attention and GQA it denotes processed key positions in the online state. The output view extent and tile size must agree with the declared loop.

### Tile-size evidence

`tile_axes[].size_ref` is an optional evidence key for the source constant behind a tile size (for example the `constexpr int BM` line). The validator checks that the ref exists and that its literal line matches; it cannot prove that the scenario's `params` equal the source's compile-time constants. Tile sizes and sample shapes remain a human-review item: compare `size` and any `size_ref` against the cited constant and the launcher, and treat a model as unreviewed until that comparison is recorded.

### Limits

The validator keeps the browser demo bounded and rejects a model that exceeds these limits:

| Resource | Limit |
| --- | --- |
| Output tiles per scenario | 256 |
| Expanded steps per scenario | 96 |
| Worker grid axis (`x`, `y`, `z`) | 1..256 |
| Claims per PID | 1..256; worker count × claims must be at most 256 |
| Sample variables per address | 16 |
| Address sample evaluations per scenario (addresses × K phases × tiles) | 20,000 |
| Length of one formula/expression string | 1024 characters |
| Cells in one full output tile (product of axis sizes) | 1,000,000 |

The validator expands every `(worker, claim_i)` pair and requires **exactly one claim for every output tile**. Multiple tiles may belong to one PID when `claim_loop` is present; those tiles are ordered by `claim_i` and get serial dependencies. It rejects duplicate or missing tiles, out-of-range coordinates/groups, invalid source evidence, unsupported formulas, empty loop iterations, and cyclic dependencies. Running `python3 -m op_show validate <model.json>` either prints a one-line summary or a one-line `Invalid model: ...` reason. The launch/PID panel displays the expanded claims before the output matrix. The two [GQA strategy models](../examples/triton_gqa_q_owned.json) and [KV-owned variant](../examples/triton_gqa_kv_owned.json) use the same kernel file to compare one Q output task per PID with a KV-head PID looping over Q tasks.

For a 2D output, the first tile axis is displayed as rows and the second as columns. For a 1D output, tiles form one row. A tile's padded cells are determined by output bounds.

The optional `gemm_view` uses the same validated output tile bounds and K-loop phase to draw A[M tile,K slice], B[K slice,N tile], and C[M tile,N tile]. It checks that the K slice count matches the declared loop, and that source refs and highlighted step IDs exist. See [cuda_tensor_core_gemm.json](../examples/cuda_tensor_core_gemm.json). Matrix cells are a logical view; WMMA fragment register ownership is intentionally not inferred, because NVIDIA leaves that mapping unspecified.

The optional `row_k_view` reuses output row bounds and K-loop phases to draw X[row tile,K slice] → Y[row tile] for `reduce`, or A[row tile,K slice] × x[K slice] → y[row tile] for `gemv`. `source_refs` must contain `matrix`, `reduce`, and `output`; `gemv` additionally requires `vector`. `step_roles` names existing step IDs and chooses among `matrix`, `vector`, and `output`. The validator checks those refs and the K-slice count, but the author must still compare formulas with the cited source. See [cuda_warp_reduce.json](../examples/cuda_warp_reduce.json) and [cuda_warp_gemv.json](../examples/cuda_warp_gemv.json). Masked input cells are skipped by source bounds checks; masked output rows are not written.

The optional `attention_view` uses output query row and feature bounds plus each repeated key-loop phase to draw Q[query tile,D], K/V[key tile,D], logical scores S[query tile,key tile], and O[query tile,D]. `source_refs` requires `q`, `k`, `v`, `score`, `softmax`, and `o`; `step_roles` chooses among `Q`, `K`, `V`, `S`, and `O`. The validator checks source refs, step IDs, N-slice count, and a 4096-cell cap per rendered matrix. It does not prove online-softmax numerical equivalence to the source. See [triton_flash_attention2.json](../examples/triton_flash_attention2.json). S is a logical score tile, not a separately stored global tensor; padded query rows are unused, padded key columns receive −∞ before softmax.

## Safe expressions

Expressions accept integer constants, named parameters, `wid_x/y/z`, `claim_i`, `tile_<axis>`, `phase`, `+`, `-`, `*`, `//`, `%`, `ceildiv(a,b)`, `min(...)`, and `max(...)`. They reject arbitrary Python, attributes, indexing, powers, and unknown names. Scope: tile-axis `index` formulas see parameters, `wid_*`, and `claim_i`; later axes also see earlier `tile_<axis>` values; task-group `index` sees every `tile_<axis>`; address `expr`/`sample` formulas additionally see `phase`. Address `sample` variables are evaluated in their listed order; a later variable may refer to an earlier one.

Division follows Python integer semantics, not C: `//` floors toward negative infinity and `%` takes the sign of the divisor (for example `-7 // 2 == -4` and `-7 % 3 == 2`). Negative intermediate values are allowed, but a resulting address offset must be non-negative and a tile index must fall inside its axis. `ceildiv(a, b)` is mathematical ceiling division and requires `b > 0`.

## v2: source-backed logical task trace

Start from [ascend_qkv_alltoall.json](../examples/ascend_qkv_alltoall.json). This first v2 implementation supports `trace.kind: "qkv-head-grouped"` for a producer/consumer head-grouped branch with the formulas below. It is a bounded, deterministic description for one loop structure, not a parser for arbitrary CUDA or Triton source. For a new model, point `source.path` at your own local kernel and cite your own lines.

| Field | Meaning |
| --- | --- |
| `trace.kind` | `qkv-head-grouped`. |
| `trace.formulas` | Exactly the 14 formula keys listed below; each value is an integer expression. |
| `trace.formula_refs` | Exactly the same 14 keys, each mapping to an existing evidence key. |
| `trace.refs` | Exactly the 13 evidence keys listed below. |
| `trace.param_refs` | Optional map from a parameter or derived value to an existing evidence key (see below). |
| `scenarios` | `aligned` and `padded`, with the required parameters and hard constraints described below. |

### Formulas

The validator requires exactly these keys in `trace.formulas` and `trace.formula_refs`:

| Key | Meaning |
| --- | --- |
| `lane` | `flat_idx % s_wave`; block offset inside a wave. |
| `task_idx` | `flat_idx // s_wave`; flattened (target rank, head group) index. |
| `peer_rank` | Target rank, `task_idx % rank_size`. |
| `head_group` | `task_idx // rank_size`. |
| `block_id` | `wave_start + lane`. |
| `consumer_flat_idx` | Consumer-side flat index reconstructed from (head group, source rank, lane). |
| `signal_slot` | Signal slot for a remote task. |
| `signal_offset` | Signal element offset, `signal_slot * signal_stride`. |
| `output_row` | First output row for the task. |
| `peer_q` | Peer-buffer Q offset for a remote task. |
| `peer_k` | Peer-buffer K offset, `peer_q + heads_per_rank * peer_head_size`. |
| `peer_v` | Peer-buffer V offset, `peer_q + 2 * heads_per_rank * peer_head_size`. |
| `input_offset` | First input element offset for the task. |
| `output_offset` | First output element offset for the task. |

Formula variables are the 17 scenario parameters, the derived values `num_blocks`, `head_groups`, `buffer_num`, `peer_head_size`, `fused_heads`, and `block_slot_size`, and the per-task variables `source_rank`, `wave_start`, `flat_idx`, `lane`, `task_idx`, `target_rank`, `head_group`, `block_id`, `head_start`, `output_row`, `signal_slot`, and `peer_q` in that evaluation order. The validator evaluates each formula and compares it with its own expansion; a formula that evaluates to a different value is rejected even if its evidence line matches.

### `trace.refs` keys

The validator requires exactly these 13 keys. Five are validation-only: they must cite evidence, but no emitted event uses them. The other eight are attached to the events they name.

| Key | Use |
| --- | --- |
| `role` | Validation-only. Producer/consumer branch evidence. |
| `wave` | Validation-only. Wave loop evidence. |
| `producer_loop` | Validation-only. Producer loop bound evidence. |
| `consumer_loop` | Validation-only. Consumer loop bound evidence. |
| `local` | Validation-only. Local-target branch evidence. |
| `load` | Attached to `load` events. |
| `direct` | Attached to `direct` (local write) events. |
| `peer_store` | Attached to `peer_store` (remote staging) events. |
| `fence` | Attached to `fence` events. |
| `notify` | Attached to `notify` events. |
| `wait` | Attached to `wait` events. |
| `consume` | Attached to `consume` events. |
| `output_store` | Attached to `output_store` events. |

### Parameter evidence (`trace.param_refs`)

`trace.param_refs` is optional. It maps a parameter or derived value to an existing `source.refs` key, for example `{"s_wave": "s_wave", "buffer_num": "buffer_num"}`. The validator checks that every value is an existing evidence key; it does not prove that the cited line computes the value, and it does not require the parameter name to be one of the 17 scenario parameters. The bundled example wires the parameter and launcher constants this way:

```json
"param_refs": {
  "producer_cores": "role_counts",
  "s_wave": "s_wave",
  "heads_per_group": "group_size",
  "head_groups": "head_groups",
  "block_s": "block_s",
  "block_d": "block_d",
  "signal_stride": "signal_stride",
  "buffer_num": "buffer_num",
  "input_strides": "strides",
  "output_strides": "strides",
  "total_wave_tasks": "total_tasks",
  "mask": "mask"
}
```

The `strides`, `total_tasks`, and `mask` refs are doc/param evidence: they point a reviewer at the launcher stride call, the wave-task count, and the load mask predicate, but no formula or event consumes them. The validator only checks the literal source lines; reviewing the parameter values against the cited code remains a human step, exactly as for v1 `size_ref`.

### Parameters, derived values, hard constraints

`scenarios[].params` must contain these 17 positive integers: `sequence_length`, `rank_size`, `heads_per_rank`, `total_cores`, `producer_cores`, `consumer_cores`, `s_wave`, `block_s`, `block_d`, `heads_per_group`, `signal_stride`, `stride_a_s`, `stride_a_h`, `stride_a_d`, `stride_out_s`, `stride_out_h`, `stride_out_d`.

The validator derives: `num_blocks = ceildiv(sequence_length, block_s)`, `head_groups = ceildiv(heads_per_rank, heads_per_group)`, `buffer_num = num_blocks`, `wave_count = ceildiv(num_blocks, s_wave)`, `total_wave_tasks = rank_size * head_groups * s_wave`, `peer_head_size = block_s * block_d`, `fused_heads = 3 * heads_per_rank`, `block_slot_size = rank_size * fused_heads * peer_head_size`, `head_start = head_group * heads_per_group`, and `valid_tokens = min(block_s, sequence_length - block_id * block_s)`.

The validator rejects a scenario that violates any of these hard constraints:

| Constraint | Rule |
| --- | --- |
| `rank_size` | 2..16 |
| `total_cores` | 2..256 |
| Producer/consumer split | `producer_cores == (total_cores + 1) // 2` and `consumer_cores == total_cores - producer_cores` |
| `heads_per_rank` | Greater than 4 (the block-flat branch for 1..4 heads is outside this schema) |
| `heads_per_group` | `2` if `heads_per_rank <= 16`, else `4` |
| `s_wave` | `4` if `rank_size <= 6`, else `2` if `rank_size < 24`, else `1` (the `1` branch is unreachable while `rank_size <= 16`) |
| Block constants | `block_s == 112`, `block_d == 128`, `signal_stride == 16` |
| Contiguous strides | `stride_a_s == rank_size * heads_per_rank * block_d`, `stride_a_h == block_d`, `stride_a_d == 1`; `stride_out_s == heads_per_rank * block_d`, `stride_out_h == block_d`, `stride_out_d == 1` |
| Aligned/padded | `aligned` requires `sequence_length % 112 == 0` (`block_s == 112`); `padded` requires a tail block |
| Expansion caps | At most 2048 tasks and 10000 events per scenario |

### What the v2 validator does and does not check

- **Strides are forced contiguous.** The input is assumed to have shape `[sequence_length, rank_size * heads_per_rank, block_d]` and the output `[rank_size * sequence_length, heads_per_rank, block_d]` with the exact contiguous strides above. The source launcher accepts arbitrary `q.stride()` values, but this model format does not: a non-contiguous declaration is rejected. The bundled example uses contiguous tensors.
- **Mask counts are derived, not declared.** The validator computes `valid_tokens = min(block_s, sequence_length - block_id * block_s)` itself. There is no declared mask count to compare against; the padded case's 68/112 tail is a consequence of the derived value, not an independently checked claim.
- **The task layout is hardcoded in `op_show/trace.py`, from the cited source.** The producer/consumer split, group-size rule, `s_wave`, the 112/128/16 block constants, and `buffer_num = num_blocks` are implemented in Python and are not parsed out of the Ascend source. Source refs (including `param_refs`) show that matching lines exist; keeping those hardcoded rules in sync with the source is a reviewer-maintained step.
- **Events are task-level aggregates.** Each task emits at most one `load` event and one output event (`direct` for local, `peer_store` for remote). A remote task additionally gets one `notify` event from its producer and one `wait` + `consume` + `output_store` sequence from its consumer. One event covers the task's head group (`heads_per_group` heads) times Q/K/V; it is not one event per memory instruction.
- **The event graph is logical, not a device schedule.** Producer events for a PID's wave are followed by one `fence`, then the remote `notify` events; consumer `wait`/`consume`/`output_store` events follow their producer's notify and the previous output store of the same consumer PID. Local tasks have `load` and `direct` only, with no wait. Events form a dependency DAG; the viewer advances all ready events at each logical level and displays causal order, not device timing.
- **Each task has `producer_pid` and, for remote tasks, `consumer_pid`.** These are execution-unit indices, and the same PID owns many tasks within a wave and across waves. A task does not own a distinct PID, and an execution-unit ID is never used as a logical tile ID.
- **A wave is a per-PID loop slice, not a global barrier across PIDs.** Cross-PID dependencies appear only on wait/consume/output-store events.

## Agent extraction boundary

Read both the kernel and its launcher or host tiling setup. If shape or tile sizes are absent, choose small **example** dimensions that preserve the source formulas, then choose a second set with a tail. Do not claim that those sample shapes were measured or provided by the user. Every mapping, stage, address expression, and v2 formula should cite a matching source line, and v1 tile sizes plus v2 parameter/layout constants should be wired through `size_ref` or `param_refs` where the source has a stable line to cite. Literal evidence makes the model reviewable; it does not establish full semantic equivalence, and it does not prove that `params` equal the source's compile-time constants. If the source uses dynamic branching, opaque pointer arithmetic, or cross-tile synchronization that cannot be reduced to these rules, return an `unknown` model with an explanation instead of inventing an animation.

## v2 example and evidence

The bundled [v2 model](../examples/ascend_qkv_alltoall.json) cites [its local source copy](../examples/sources/ascend_qkv_all2all.py), accompanied by the [upstream permission notice](../examples/sources/TRITON_DISTRIBUTED_ASCEND_LICENSE). A new `source-backed` v2 model must cite its own operator source and lines; do not reuse the bundled line refs for another kernel. The Python tests also build a synthetic, test-only source to exercise the validator.

The format can express a 4-rank, 20-head-per-rank, 32-AIV trace with `S_WAVE=4`, `BLOCK_S=112`, and `BLOCK_D=128`; such parameters are provided by the model author and are not measured hardware values.
