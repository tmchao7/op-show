---
name: op-show
description: Create a source-backed, hardware-free tile execution walkthrough or QKV task trace for a CUDA C++, CUDA Triton, or Triton-Ascend operator. Use when asked to explain tile IDs, bounds, padding, address offsets, or ideal parallel and serial progress from local operator source.
---

# OP/SHOW agent workflow

1. Read the user's kernel source, launcher, and tiling constants. Start with the launch grid, every PID axis, task-claim loops, and the exact `(PID, claim_i) → logical output tile` mapping. Then identify output dimensions, tile sizes, shared input groups, masks, ordered stages, and any cross-tile dependencies. Do not run untrusted kernel code.
2. For a task-routing kernel such as QKV All-to-All, inspect Producer and Consumer loops, PID assignment, task flat indices, local versus remote branches, peer_mem addresses, signal slots, and fence/notify/wait ordering. Use v2 only when its `qkv-head-grouped` formulas fit the source. Otherwise use v1 for simple output tiles or mark the model `unknown`.
3. Locate the OP/SHOW checkout that contains this skill (follow the skill-directory symlink if needed). Read its `docs/model-format.md` and copy the closest `examples/*.json` as a starting point. Set `source.path` to the real local file and add exact line refs for every modeled rule.
4. Keep worker ID separate from logical tile ID. If a PID claims several output tasks, use v1 `claim_loop` only when a bounded, uniform claim count and deterministic formula fit the source; cite every nested loop. Use `parallelism.task_group` to show shared inputs such as GQA's KV head, but never infer cache residency or speedup from shared ownership. Compare the Q-owned and KV-owned GQA examples when explaining that choice. Use only the format's integer formula language. Include an aligned small shape and a second shape with a tail tile; if a reduction axis is tiled, make its tail visible too when possible. Label dimensions as chosen examples when they were not supplied.
5. If a rule cannot be traced to source, mark the mapping `unknown` and explain the uncertainty. Never turn a guessed scheduling order, dynamic branch, or opaque pointer calculation into a definite animation.
6. Run `python3 -m op_show validate <model.json>` (or `op-show validate`) and fix every error. Then run `python3 -m op_show serve <model.json>` and give the user the printed local URL. Review launch/PID count, claim order, output coverage, shared-input group, tile IDs, bounds, masks, address samples, and stage order against the source before reporting success.

For v2, test an example task end to end: `flat_idx → Producer PID → peer_mem → signal slot → Consumer PID → qkv_out`, and confirm the final block mask. Verify that local direct writes have no wait and that each Producer fences after its own wave task loop before its notify loop. Wave is not a global PID barrier.

The viewer shows an **ideal logical schedule**. Independent tiles advance together; it does not model SM/NPU capacity, stalls, latency, or physical launch order. The validator checks internal consistency and literal source evidence, not equivalence for arbitrary GPU programs.
