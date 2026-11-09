---
name: op-show
description: Create a source-backed, hardware-free tile execution walkthrough for a CUDA C++, CUDA Triton, or Triton-Ascend operator. Use when asked to explain tile IDs, bounds, padding, address offsets, or ideal parallel and serial progress from local operator source.
---

# OP/SHOW agent workflow

1. Read the user's kernel source, launcher, and tiling constants. Identify output dimensions, tile sizes, worker/program IDs, output ownership, masks, ordered stages, and any cross-tile dependencies. Do not run untrusted kernel code.
2. Locate the OP/SHOW checkout that contains this skill (follow the skill-directory symlink if needed). Read its `docs/model-format.md` and copy the closest `examples/*.json` as a starting point. Set `source.path` to the real local file and add exact line refs for every modeled rule.
3. Keep worker ID separate from logical tile ID. Use only the format's integer formula language. Include an aligned small shape and a second shape with a tail tile; if a reduction axis is tiled, make its tail visible too when possible. Label dimensions as chosen examples when they were not supplied.
4. If a rule cannot be traced to source, mark the mapping `unknown` and explain the uncertainty. Never turn a guessed scheduling order, dynamic branch, or opaque pointer calculation into a definite animation.
5. Run `python3 -m op_show validate <model.json>` (or `op-show validate`) and fix every error. Then run `python3 -m op_show serve <model.json>` and give the user the printed local URL. Review tile IDs, bounds, masks, address samples, and stage order against the source before reporting success.

The viewer shows an **ideal logical schedule**. Independent tiles advance together; it does not model SM/NPU capacity, stalls, latency, or physical launch order. The validator checks internal consistency and literal source evidence, not equivalence for arbitrary GPU programs.
