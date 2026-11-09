# OP/SHOW

OP/SHOW turns a **source-backed tile model** into a local, hardware-free execution walkthrough. It shows tile IDs, output bounds, masked tail cells, example address offsets, and which tile stages can advance in parallel. It does not run the kernel or estimate device timing.

## Try the demos

Python 3.9+ is enough; there are no runtime dependencies.

```sh
python3 -m op_show validate examples/cuda_tiled_gemm.json
python3 -m op_show serve examples --no-browser
```

Open the printed local URL. The operator menu switches between native CUDA C++ GEMM, CUDA Triton matmul, a Triton-Ascend vector add, and the v2 QKV All-to-All task trace. Each has an aligned case and a tail/padding case. Omit `--no-browser` to open the browser automatically.

The QKV trace links four views: Producer block × target rank/head group, PID claims, a peer_mem path, and Consumer source rank/block × head group output. Select a task to follow its Q/K/V offsets, signal slot, source lines, and logical event order. The example uses four ranks and 32 AIVs with `S=1344` or `S=1300`; these are illustrative inputs, not measured hardware traces. The second case masks 44 tokens in the final 112-token block.

The viewer opens in Chinese and has a 中文 / EN switch in the header. The choice is saved in this browser. The included demos provide translated titles, steps, and address labels; custom models can add the optional Chinese fields described in [the format guide](docs/model-format.md).

The equivalent installed command is `op-show` after `python3 -m pip install -e .`. The repository also includes the executable `./op-show` wrapper.

```sh
op-show validate path/to/model.json
op-show serve path/to/model.json
op-show serve examples --port 8765
```

`serve` only binds to `127.0.0.1` by default. It serves the validated model and its local source file read-only. The browser never executes the operator code. No cloud API, GPU, NPU, or JavaScript build step is needed.

## Bring your own kernel

An agent can read a local source file and create a model using [the format guide](docs/model-format.md). Start from the closest JSON under `examples/`, replace its source path and refs, then set the mapping and two small scenarios. Run `op-show validate` before opening the viewer.

The portable agent instructions are in [skills/op-show/SKILL.md](skills/op-show/SKILL.md). Link that directory into an agent's skill directory so its references to this checkout stay valid; agents without skill support can follow the same CLI and format guide. The CLI does not call a model itself.

For follow-on development, read [AGENTS.md](AGENTS.md) and use the copyable [agent handoff prompt](docs/AGENT_HANDOFF.md).

The validator checks literal source line references, safe integer formulas, worker-to-tile coverage, bounds, padding, address samples, and dependency cycles. For the v2 QKV format it also checks flat task IDs, PID claims, peer/output offsets, signal slots, full output coverage, and notify/wait pairing. **It cannot prove that an arbitrary CUDA or Triton program is equivalent to an agent's description.** Only use `source-backed` for simple, explicit mappings you can inspect. For dynamic or uncertain mappings, use `"status": "unknown"` with a note; the page disables playback.

## Ideal execution model

- Each declared worker (CUDA block or Triton program) owns one logical output tile in v1. IDs remain separate so their meanings are visible.
- All tiles without dependencies start together. A tile's stages execute in source order; repeated K slices are serial within that tile.
- Explicit cross-tile dependencies delay only their target tiles. There is no SM/core capacity, block scheduling order, memory stall, or elapsed-time model.
- The percentage is completed tile-stage work, not measured utilization or throughput.

In v2, each tick completes every event whose dependencies have been satisfied at that DAG level. A Producer PID finishes its wave's owned tasks, fences, then notifies remote targets. A Consumer waits and consumes before writing Q/K/V. Local targets write directly and have no wait. Wave is a loop slice for each PID, not a global barrier across PIDs.

## Checks

```sh
python3 -m unittest discover -s tests -v
node --check op_show/web/app.js
```

The example CUDA/Triton sources are readable illustrations of the declared mapping. They are not compiled or executed in this hardware-free environment.
