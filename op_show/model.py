"""Validate and expand a small source-backed tile description.

This checks the declared model, coverage, and literal source evidence. It does
not prove that arbitrary CUDA/Triton source is semantically equivalent to the
agent's description; the browser makes that boundary explicit.
"""

import itertools
import json
from pathlib import Path

from .expr import ExpressionError, evaluate


class ValidationError(ValueError):
    pass


BACKENDS = {"cuda", "triton-cuda", "triton-ascend"}
MAX_TILES = 256
MAX_STEPS = 96


def _require(condition, message):
    if not condition:
        raise ValidationError(message)


def _localized(document, field, context):
    value = document.get(field)
    if value is not None:
        _require(isinstance(value, str) and value.strip(), f"{context}.{field} must be a nonempty string")
    return value


def _number(expression, variables, context):
    try:
        return evaluate(expression, variables)
    except ExpressionError as exc:
        raise ValidationError(f"{context}: {exc}") from exc


def _source_and_refs(document, model_path):
    source = document.get("source")
    _require(isinstance(source, dict), "source must be an object")
    rel_path = source.get("path")
    _require(isinstance(rel_path, str) and rel_path, "source.path is required")
    source_path = (model_path.parent / rel_path).resolve()
    _require(source_path.is_file(), f"source file not found: {source_path}")
    _require(isinstance(source.get("kernel"), str) and source["kernel"], "source.kernel is required")
    source_text = source_path.read_text(encoding="utf-8")
    _require(source["kernel"] in source_text, f"kernel name {source['kernel']!r} not found in source")
    refs = source.get("refs")
    _require(isinstance(refs, dict) and refs, "source.refs must contain source evidence")
    lines = source_text.splitlines()
    checked = {}
    for key, ref in refs.items():
        _require(isinstance(key, str) and key, "source ref keys must be nonempty strings")
        _require(isinstance(ref, dict), f"source.refs.{key} must be an object")
        line, fragment = ref.get("line"), ref.get("contains")
        _require(type(line) is int and 1 <= line <= len(lines), f"source.refs.{key}: invalid line")
        _require(isinstance(fragment, str) and fragment.strip(), f"source.refs.{key}: contains is required")
        _require(fragment in lines[line - 1], f"source.refs.{key}: text does not match line {line}")
        checked[key] = {"line": line, "contains": fragment}
    return source_text, checked


def _evidence(key, refs, context):
    _require(isinstance(key, str) and key in refs, f"{context}: missing source evidence ref")
    return key


def _steps(program, params, refs):
    _require(isinstance(program, list) and program, "program must contain steps")
    result = []
    for index, item in enumerate(program):
        _require(isinstance(item, dict) and isinstance(item.get("steps"), list) and item["steps"],
                 f"program[{index}].steps must be a nonempty list")
        repeat = _number(item.get("repeat", "1"), params, f"program[{index}].repeat")
        _require(1 <= repeat <= MAX_STEPS, f"program[{index}].repeat must be 1..{MAX_STEPS}")
        loop_axis, loop_tile = item.get("loop_axis"), item.get("loop_tile")
        if repeat > 1:
            _require(isinstance(loop_axis, str) and loop_axis in params, f"program[{index}]: loop_axis is required")
            _require(isinstance(loop_tile, str) and loop_tile in params, f"program[{index}]: loop_tile is required")
            _evidence(item.get("loop_source_ref"), refs, f"program[{index}].loop")
        for iteration in range(repeat):
            loop_valid = None
            if repeat > 1:
                loop_valid = max(0, min(params[loop_tile], params[loop_axis] - iteration * params[loop_tile]))
                _require(loop_valid > 0, f"program[{index}]: empty loop iteration")
            for step in item["steps"]:
                _require(isinstance(step, dict) and isinstance(step.get("id"), str) and step["id"],
                         f"program[{index}]: step id is required")
                _require(isinstance(step.get("label"), str) and step["label"],
                         f"program[{index}]: step label is required")
                label_zh = _localized(step, "label_zh", f"program[{index}].{step['id']}")
                ref = _evidence(step.get("source_ref"), refs, f"program[{index}].{step['id']}")
                result.append({"id": step["id"], "label": step["label"], "label_zh": label_zh,
                               "source_ref": ref,
                               "iteration": iteration if repeat > 1 else None,
                               "loop_axis": loop_axis, "loop_tile": loop_tile, "loop_valid": loop_valid})
    _require(len(result) <= MAX_STEPS, f"expanded program exceeds {MAX_STEPS} steps")
    return result


def _address_samples(addresses, env, refs, phase_count):
    _require(isinstance(addresses, list), "mapping.addresses must be a list")
    result = []
    for i, address in enumerate(addresses):
        _require(isinstance(address, dict), f"address[{i}] must be an object")
        label, expr, samples = address.get("label"), address.get("expr"), address.get("sample")
        _require(isinstance(label, str) and label, f"address[{i}].label is required")
        label_zh = _localized(address, "label_zh", f"address[{i}]")
        _require(isinstance(samples, dict), f"address[{i}].sample must be an object")
        ref = _evidence(address.get("source_ref"), refs, f"address[{i}]")
        values = []
        for phase in range(phase_count):
            vars_for_address = dict(env, phase=phase)
            substitutions = {}
            for name, sample_expr in samples.items():
                _require(isinstance(name, str) and name.isidentifier(), f"address[{i}]: invalid sample variable")
                value = _number(sample_expr, vars_for_address, f"address[{i}].sample.{name}")
                vars_for_address[name] = value
                substitutions[name] = value
            offset = _number(expr, vars_for_address, f"address[{i}].expr")
            _require(offset >= 0, f"address[{i}]: negative offset")
            values.append({"phase": phase, "variables": substitutions, "offset": offset})
        result.append({"label": label, "label_zh": label_zh, "expr": expr, "source_ref": ref, "samples": values})
    return result


def _dependencies(declarations, tile_lookup, axis_count):
    _require(isinstance(declarations, list), "dependencies must be a list")
    for entry in declarations:
        _require(isinstance(entry, dict), "dependency entries must be objects")
        target = entry.get("tile")
        predecessors = entry.get("on")
        _require(isinstance(target, list) and len(target) == axis_count and all(type(x) is int for x in target),
                 "dependency tile must be a tile coordinate array")
        _require(isinstance(predecessors, list), "dependency on must be an array")
        target_key = tuple(target)
        _require(target_key in tile_lookup, f"dependency target {target} does not exist")
        for predecessor in predecessors:
            _require(isinstance(predecessor, list) and len(predecessor) == axis_count and
                     all(type(x) is int for x in predecessor), "dependency predecessor must be a tile coordinate array")
            key = tuple(predecessor)
            _require(key in tile_lookup, f"dependency predecessor {predecessor} does not exist")
            _require(key != target_key, "tile cannot depend on itself")
            tile_lookup[target_key]["dependencies"].append(list(key))

    visiting, visited = set(), set()

    def finish(key):
        if key in visiting:
            raise ValidationError("tile dependencies contain a cycle")
        if key in visited:
            return tile_lookup[key]["end"]
        visiting.add(key)
        tile = tile_lookup[key]
        tile["start"] = max((finish(tuple(dep)) for dep in tile["dependencies"]), default=0)
        tile["end"] = tile["start"] + tile["step_count"]
        visiting.remove(key)
        visited.add(key)
        return tile["end"]

    for key in tile_lookup:
        finish(key)


def validate_model(path):
    model_path = Path(path).resolve()
    try:
        document = json.loads(model_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read model: {exc}") from exc
    _require(isinstance(document, dict), "model root must be an object")
    if document.get("schema_version") == 2:
        from .trace import validate_trace_model
        return validate_trace_model(document, model_path)
    _require(document.get("schema_version") == 1, "schema_version must be 1")
    _require(document.get("backend") in BACKENDS, f"backend must be one of {', '.join(sorted(BACKENDS))}")
    _require(isinstance(document.get("title"), str) and document["title"], "title is required")
    title_zh = _localized(document, "title_zh", "model")
    source_text, refs = _source_and_refs(document, model_path)
    mapping = document.get("mapping")
    _require(isinstance(mapping, dict), "mapping must be an object")
    status = mapping.get("status")
    _require(status in ("source-backed", "unknown"), "mapping.status must be source-backed or unknown")
    base = {"schema_version": 1, "title": document["title"], "title_zh": title_zh,
            "backend": document["backend"],
            "kernel": document["source"]["kernel"], "source_path": document["source"]["path"],
            "refs": refs, "status": status, "note": mapping.get("note", ""),
            "note_zh": _localized(mapping, "note_zh", "mapping"), "scenarios": []}
    if status == "unknown":
        _require(isinstance(mapping.get("note"), str) and mapping["note"].strip(),
                 "unknown mapping requires an explanation")
        return base, source_text

    axes = mapping.get("tile_axes")
    _require(isinstance(axes, list) and 1 <= len(axes) <= 2, "mapping.tile_axes must have 1 or 2 axes")
    names = [a.get("name") for a in axes if isinstance(a, dict)]
    _require(len(names) == len(axes) and len(set(names)) == len(names) and
             all(isinstance(n, str) and n.isidentifier() for n in names), "tile axis names must be unique identifiers")
    grid = mapping.get("worker_grid")
    _require(isinstance(grid, dict) and "x" in grid, "mapping.worker_grid.x is required")
    grid_ref = _evidence(mapping.get("grid_source_ref"), refs, "worker grid")
    base["grid_source_ref"] = grid_ref
    base["worker_labels"] = mapping.get("worker_labels", {})
    scenarios = document.get("scenarios")
    _require(isinstance(scenarios, list) and len(scenarios) == 2,
             "exactly two scenarios (aligned and padded) are required")
    _require({s.get("id") for s in scenarios if isinstance(s, dict)} == {"aligned", "padded"},
             "scenario ids must be aligned and padded")

    for scenario in scenarios:
        scenario_id = scenario["id"]
        label_zh = _localized(scenario, "label_zh", scenario_id)
        params = scenario.get("params")
        _require(isinstance(params, dict) and params, f"{scenario_id}.params is required")
        _require(all(isinstance(k, str) and k.isidentifier() and type(v) is int and 1 <= v <= 100000
                     for k, v in params.items()), f"{scenario_id}: params must be positive integers")
        dimensions = []
        for i, axis in enumerate(axes):
            _require(isinstance(axis, dict), f"tile_axes[{i}] must be an object")
            name = axis["name"]
            extent = _number(axis.get("extent"), params, f"tile_axes[{i}].extent")
            size = _number(axis.get("size"), params, f"tile_axes[{i}].size")
            _require(extent > 0 and size > 0, f"{scenario_id}.{name}: extent and size must be positive")
            _evidence(axis.get("source_ref"), refs, f"tile_axes[{i}]")
            dimensions.append({"name": name, "extent": extent, "size": size,
                               "count": (extent + size - 1) // size, "index": axis.get("index"),
                               "source_ref": axis["source_ref"]})
        has_tail = any(d["extent"] % d["size"] for d in dimensions)
        _require(has_tail == (scenario_id == "padded"),
                 f"{scenario_id}: aligned must divide evenly and padded must contain a tail tile")

        grid_sizes = {}
        for key in ("x", "y", "z"):
            grid_sizes[key] = _number(grid.get(key, "1"), params, f"{scenario_id}.worker_grid.{key}")
            _require(1 <= grid_sizes[key] <= MAX_TILES, f"{scenario_id}: worker grid {key} outside 1..{MAX_TILES}")
        worker_count = grid_sizes["x"] * grid_sizes["y"] * grid_sizes["z"]
        expected_count = 1
        for dimension in dimensions:
            expected_count *= dimension["count"]
        _require(worker_count <= MAX_TILES and expected_count <= MAX_TILES,
                 f"{scenario_id}: demo supports at most {MAX_TILES} tiles")
        steps = _steps(mapping.get("program"), params, refs)
        phase_count = max((step["iteration"] for step in steps if step["iteration"] is not None), default=0) + 1
        tiles = []
        lookup = {}
        for z, y, x in itertools.product(range(grid_sizes["z"]), range(grid_sizes["y"]), range(grid_sizes["x"])):
            env = dict(params, wid_x=x, wid_y=y, wid_z=z)
            coords, bounds = [], []
            for dimension in dimensions:
                index = _number(dimension["index"], env, f"{scenario_id}.{dimension['name']}.index")
                _require(0 <= index < dimension["count"],
                         f"{scenario_id}: worker ({x},{y},{z}) maps outside tile axis {dimension['name']}")
                env[f"tile_{dimension['name']}"] = index
                coords.append(index)
                origin = index * dimension["size"]
                bounds.append({"axis": dimension["name"], "start": origin,
                               "end": min(origin + dimension["size"], dimension["extent"]),
                               "tile_end": origin + dimension["size"]})
            key = tuple(coords)
            _require(key not in lookup, f"{scenario_id}: tile {coords} is covered by multiple workers")
            valid_cells = 1
            total_cells = 1
            for dimension, bound in zip(dimensions, bounds):
                valid_cells *= bound["end"] - bound["start"]
                total_cells *= dimension["size"]
            tile = {"coords": coords, "worker": {"x": x, "y": y, "z": z}, "bounds": bounds,
                    "valid_cells": valid_cells, "total_cells": total_cells,
                    "addresses": _address_samples(mapping.get("addresses", []), env, refs, phase_count),
                    "dependencies": [], "step_count": len(steps)}
            tiles.append(tile)
            lookup[key] = tile
        expected = set(itertools.product(*(range(d["count"]) for d in dimensions)))
        missing = expected - set(lookup)
        _require(not missing, f"{scenario_id}: missing tile coverage: {sorted(missing)[:4]}")
        _dependencies(mapping.get("dependencies", []), lookup, len(dimensions))
        base["scenarios"].append({"id": scenario_id, "label": scenario.get("label", scenario_id),
                                   "label_zh": label_zh,
                                   "params": params, "dimensions": dimensions, "grid": grid_sizes,
                                   "steps": steps, "tiles": tiles,
                                   "total_ticks": max(tile["end"] for tile in tiles)})
    return base, source_text
