"""Expand and check the source-backed QKV head-grouped logical task trace.

The input contains bounded example parameters and cited integer formulas. The
expanded event DAG is a logical partial order, never a device timing trace.
"""

import itertools

from .model import (
    BACKENDS, ValidationError, _evidence, _localized, _number, _require,
    _source_and_refs,
)


MAX_TRACE_TASKS = 2048
MAX_TRACE_EVENTS = 10000
FORMULA_KEYS = {
    "lane", "task_idx", "peer_rank", "head_group", "block_id",
    "consumer_flat_idx", "signal_slot", "signal_offset", "output_row",
    "peer_q", "peer_k", "peer_v", "input_offset", "output_offset",
}
EVENT_REFS = {
    "role", "wave", "producer_loop", "consumer_loop", "load", "local",
    "direct", "peer_store", "fence", "notify", "wait", "consume",
    "output_store",
}
PARAM_KEYS = {
    "sequence_length", "rank_size", "heads_per_rank", "total_cores",
    "producer_cores", "consumer_cores", "s_wave", "block_s", "block_d",
    "heads_per_group", "signal_stride",
    "stride_a_s", "stride_a_h", "stride_a_d",
    "stride_out_s", "stride_out_h", "stride_out_d",
}


def _roles(total, producer_count):
    producers, consumers = {}, {}
    for pid in range(total):
        before = (pid * producer_count + total - 1) // total
        through = ((pid + 1) * producer_count + total - 1) // total
        if through > before:
            _require(through - before == 1 and before not in producers,
                     "producer PID mapping is not one-to-one")
            producers[before] = pid
        else:
            logical = pid - before
            _require(logical not in consumers, "consumer PID mapping is not one-to-one")
            consumers[logical] = pid
    _require(len(producers) == producer_count and len(consumers) == total - producer_count,
             "producer/consumer PID coverage is incomplete")
    return producers, consumers


def _checked_formula(formulas, key, env, expected, context):
    actual = _number(formulas[key], env, f"{context}.{key}")
    _require(actual == expected,
             f"{context}.{key}: got {actual}, expected {expected} from the task layout")
    return actual


def _scenario(document, scenario, formulas, formula_refs, event_refs):
    name = scenario["id"]
    params = scenario.get("params")
    _require(isinstance(params, dict) and PARAM_KEYS <= params.keys(),
             f"{name}.params must contain {', '.join(sorted(PARAM_KEYS))}")
    _require(all(type(value) is int and value > 0 for value in params.values()),
             f"{name}.params must be positive integers")
    p = params
    s, rank_size, hpr = p["sequence_length"], p["rank_size"], p["heads_per_rank"]
    total, producer_count, consumer_count = (
        p["total_cores"], p["producer_cores"], p["consumer_cores"])
    wave_size, block_s, block_d = p["s_wave"], p["block_s"], p["block_d"]
    group_size, signal_stride = p["heads_per_group"], p["signal_stride"]
    _require(2 <= total <= 256 and rank_size <= 16, f"{name}: core or rank count is outside demo limits")
    _require(producer_count == (total + 1) // 2 and consumer_count == total - producer_count,
             f"{name}: producer/consumer core split does not match source")
    _require(hpr > 4 and group_size == (2 if hpr <= 16 else 4),
             f"{name}: heads_per_group does not match the head-grouped branch")
    _require(wave_size == (4 if rank_size <= 6 else 2 if rank_size < 24 else 1),
             f"{name}: s_wave does not match launcher")
    _require(block_s == 112 and block_d == 128 and signal_stride == 16,
             f"{name}: block or signal constants do not match cited source")
    _require((p["stride_a_s"], p["stride_a_h"], p["stride_a_d"]) ==
             (rank_size * hpr * block_d, block_d, 1) and
             (p["stride_out_s"], p["stride_out_h"], p["stride_out_d"]) ==
             (hpr * block_d, block_d, 1),
             f"{name}: illustrative contiguous input/output strides are inconsistent with shape")
    _require((s % block_s == 0) == (name == "aligned"),
             f"{name}: aligned must divide evenly and padded must contain a tail")

    num_blocks = (s + block_s - 1) // block_s
    head_groups = (hpr + group_size - 1) // group_size
    buffer_num = num_blocks
    wave_count = (num_blocks + wave_size - 1) // wave_size
    total_wave_tasks = rank_size * head_groups * wave_size
    task_count = rank_size * rank_size * num_blocks * head_groups
    _require(task_count <= MAX_TRACE_TASKS,
             f"{name}: expanded trace exceeds {MAX_TRACE_TASKS} tasks")
    producer_pids, consumer_pids = _roles(total, producer_count)
    peer_head_size = block_s * block_d
    fused_heads = 3 * hpr
    block_slot_size = rank_size * fused_heads * peer_head_size
    stride_a_s, stride_a_h = p["stride_a_s"], p["stride_a_h"]
    stride_out_s, stride_out_h = p["stride_out_s"], p["stride_out_h"]
    common = dict(p, num_blocks=num_blocks, head_groups=head_groups, buffer_num=buffer_num,
                  peer_head_size=peer_head_size, fused_heads=fused_heads,
                  block_slot_size=block_slot_size, stride_a_s=stride_a_s,
                  stride_a_h=stride_a_h, stride_out_s=stride_out_s,
                  stride_out_h=stride_out_h)

    tasks, task_lookup = [], {}
    producer_schedule = {}
    consumer_schedule = {}
    for source_rank, wave in itertools.product(range(rank_size), range(wave_count)):
        wave_start = wave * wave_size
        for flat_idx in range(total_wave_tasks):
            env = dict(common, source_rank=source_rank, wave_start=wave_start, flat_idx=flat_idx)
            context = f"{name}.source{source_rank}.wave{wave}.flat{flat_idx}"
            lane = _checked_formula(formulas, "lane", env, flat_idx % wave_size, context)
            env["lane"] = lane
            task_idx = _checked_formula(formulas, "task_idx", env, flat_idx // wave_size, context)
            env["task_idx"] = task_idx
            target_rank = _checked_formula(formulas, "peer_rank", env, task_idx % rank_size, context)
            env["target_rank"] = target_rank
            group = _checked_formula(formulas, "head_group", env, task_idx // rank_size, context)
            env["head_group"] = group
            block_id = _checked_formula(formulas, "block_id", env, wave_start + lane, context)
            if block_id >= num_blocks:
                continue
            env["block_id"] = block_id
            head_start = group * group_size
            head_end = min(head_start + group_size, hpr)
            env["head_start"] = head_start
            consumer_flat = group * rank_size * wave_size + source_rank * wave_size + lane
            _checked_formula(formulas, "consumer_flat_idx", env, consumer_flat, context)
            valid_tokens = min(block_s, s - block_id * block_s)
            _require(valid_tokens > 0, f"{context}: empty tile")
            output_row = source_rank * s + block_id * block_s
            _checked_formula(formulas, "output_row", env, output_row, context)
            env["output_row"] = output_row
            signal_slot = ((block_id % buffer_num) * rank_size + source_rank) * head_groups + group
            _checked_formula(formulas, "signal_slot", env, signal_slot, context)
            env["signal_slot"] = signal_slot
            signal_offset = signal_slot * signal_stride
            _checked_formula(formulas, "signal_offset", env, signal_offset, context)
            peer_q = ((block_id % buffer_num) * block_slot_size
                      + source_rank * fused_heads * peer_head_size
                      + head_start * peer_head_size)
            _checked_formula(formulas, "peer_q", env, peer_q, context)
            env["peer_q"] = peer_q
            peer_k = peer_q + hpr * peer_head_size
            peer_v = peer_q + 2 * hpr * peer_head_size
            _checked_formula(formulas, "peer_k", env, peer_k, context)
            _checked_formula(formulas, "peer_v", env, peer_v, context)
            input_offset = block_id * block_s * stride_a_s + (
                target_rank * hpr + head_start) * stride_a_h
            output_offset = output_row * stride_out_s + head_start * stride_out_h
            _checked_formula(formulas, "input_offset", env, input_offset, context)
            _checked_formula(formulas, "output_offset", env, output_offset, context)

            producer_logical = flat_idx % producer_count
            consumer_logical = consumer_flat % consumer_count
            route = "local" if source_rank == target_rank else "remote"
            task_id = f"s{source_rank}-t{target_rank}-b{block_id}-g{group}"
            _require(task_id not in task_lookup, f"{name}: duplicate output tile {task_id}")
            task = {
                "id": task_id, "source_rank": source_rank, "target_rank": target_rank,
                "block_id": block_id, "head_group": group, "head_start": head_start,
                "head_end": head_end, "wave": wave, "lane": lane,
                "flat_idx": flat_idx, "producer_logical": producer_logical,
                "producer_pid": producer_pids[producer_logical],
                "consumer_flat_idx": consumer_flat if route == "remote" else None,
                "consumer_logical": consumer_logical if route == "remote" else None,
                "consumer_pid": consumer_pids[consumer_logical] if route == "remote" else None,
                "route": route, "valid_tokens": valid_tokens, "total_tokens": block_s,
                "input_offset": input_offset, "output_row": output_row,
                "output_offset": output_offset,
                "peer_offsets": {"q": peer_q, "k": peer_k, "v": peer_v} if route == "remote" else None,
                "signal_slot": signal_slot if route == "remote" else None,
                "signal_offset": signal_offset if route == "remote" else None,
                "event_ids": [],
            }
            tasks.append(task)
            task_lookup[task_id] = task
            producer_schedule.setdefault((source_rank, wave, producer_logical), []).append(task)
            if route == "remote":
                consumer_schedule.setdefault((target_rank, wave, consumer_logical), []).append(task)

    expected = {(src, dst, block, group)
                for src, dst, block, group in itertools.product(
                    range(rank_size), range(rank_size), range(num_blocks), range(head_groups))}
    actual = {(t["source_rank"], t["target_rank"], t["block_id"], t["head_group"]) for t in tasks}
    _require(actual == expected, f"{name}: missing or overlapping output tile coverage")

    events, by_id = [], {}

    def emit(event_id, kind, task, rank, pid, wave, deps):
        _require(event_id not in by_id, f"{name}: duplicate event {event_id}")
        _require(all(dep in by_id for dep in deps), f"{name}: event {event_id} has unknown dependency")
        event = {
            "id": event_id, "kind": kind, "task_id": task["id"] if task else None,
            "rank": rank, "pid": pid, "wave": wave,
            "depends_on": list(dict.fromkeys(deps)), "source_ref": event_refs[kind],
        }
        event["level"] = max((by_id[dep]["level"] for dep in event["depends_on"]), default=-1) + 1
        events.append(event)
        by_id[event_id] = event
        if task:
            task["event_ids"].append(event_id)
        return event_id

    last_producer = {}
    for source_rank, wave in itertools.product(range(rank_size), range(wave_count)):
        for logical in range(producer_count):
            owned = producer_schedule.get((source_rank, wave, logical), [])
            pid = producer_pids[logical]
            prior = last_producer.get((source_rank, logical))
            for task in owned:
                load = emit(f"{task['id']}:load", "load", task, source_rank, pid, wave,
                            [prior] if prior else [])
                kind = "direct" if task["route"] == "local" else "peer_store"
                prior = emit(f"{task['id']}:{kind}", kind, task, source_rank, pid, wave, [load])
            fence = emit(f"s{source_rank}-p{logical}-w{wave}:fence", "fence", None,
                         source_rank, pid, wave, [prior] if prior else [])
            for task in owned:
                task["fence_id"] = fence
            prior = fence
            for task in owned:
                if task["route"] == "remote":
                    prior = emit(f"{task['id']}:notify", "notify", task, source_rank,
                                 pid, wave, [prior])
            last_producer[(source_rank, logical)] = prior

    last_consumer = {}
    for target_rank, wave in itertools.product(range(rank_size), range(wave_count)):
        for logical in range(consumer_count):
            owned = sorted(consumer_schedule.get((target_rank, wave, logical), []),
                           key=lambda task: task["consumer_flat_idx"])
            pid = consumer_pids[logical]
            prior = last_consumer.get((target_rank, logical))
            for task in owned:
                deps = [f"{task['id']}:notify"]
                if prior:
                    deps.append(prior)
                wait = emit(f"{task['id']}:wait", "wait", task, target_rank, pid, wave, deps)
                consume = emit(f"{task['id']}:consume", "consume", task,
                               target_rank, pid, wave, [wait])
                prior = emit(f"{task['id']}:output_store", "output_store", task,
                             target_rank, pid, wave, [consume])
            if prior:
                last_consumer[(target_rank, logical)] = prior

    _require(len(events) <= MAX_TRACE_EVENTS,
             f"{name}: expanded trace exceeds {MAX_TRACE_EVENTS} events")
    remote = {task["id"] for task in tasks if task["route"] == "remote"}
    for kind in ("notify", "wait", "consume"):
        _require({event["task_id"] for event in events if event["kind"] == kind} == remote,
                 f"{name}: {kind} must occur exactly once for every remote tile")
    _require(len([event for event in events if event["kind"] == "output_store"]) == len(remote),
             f"{name}: remote output coverage is incomplete")
    _require(len([event for event in events if event["kind"] == "direct"]) ==
             len(tasks) - len(remote), f"{name}: local output coverage is incomplete")

    return {
        "id": name, "label": scenario.get("label", name),
        "label_zh": _localized(scenario, "label_zh", name),
        "params": params, "num_blocks": num_blocks, "head_groups": head_groups,
        "wave_count": wave_count, "waves": [
            {"index": wave, "block_start": wave * wave_size,
             "block_end": min((wave + 1) * wave_size, num_blocks)}
            for wave in range(wave_count)],
        "spaces": {
            "input": [s, rank_size * hpr, block_d],
            "peer_mem": [buffer_num, rank_size, fused_heads, block_s, block_d],
            "output": [rank_size * s, hpr, block_d],
        },
        "producer_pids": [producer_pids[i] for i in range(producer_count)],
        "consumer_pids": [consumer_pids[i] for i in range(consumer_count)],
        "formulas": formulas, "formula_refs": formula_refs,
        "tasks": tasks, "events": events,
        "total_ticks": max(event["level"] for event in events) + 1,
    }


def validate_trace_model(document, model_path):
    _require(document.get("backend") in BACKENDS, "v2 backend is not supported")
    _require(isinstance(document.get("title"), str) and document["title"], "title is required")
    title_zh = _localized(document, "title_zh", "model")
    source_text, refs = _source_and_refs(document, model_path)
    mapping = document.get("mapping")
    _require(isinstance(mapping, dict), "mapping must be an object")
    status = mapping.get("status")
    _require(status in ("source-backed", "unknown"), "mapping.status must be source-backed or unknown")
    base = {
        "schema_version": 2, "kind": "trace", "title": document["title"],
        "title_zh": title_zh, "backend": document["backend"],
        "kernel": document["source"]["kernel"], "source_path": document["source"]["path"],
        "refs": refs, "status": status, "note": mapping.get("note", ""),
        "note_zh": _localized(mapping, "note_zh", "mapping"),
        "scenarios": [],
    }
    if status == "unknown":
        _require(isinstance(mapping.get("note"), str) and mapping["note"].strip(),
                 "unknown mapping requires an explanation")
        return base, source_text

    trace = document.get("trace")
    _require(isinstance(trace, dict) and trace.get("kind") == "qkv-head-grouped",
             "v2 source-backed trace.kind must be qkv-head-grouped")
    base["trace_kind"] = trace["kind"]
    formulas, formula_refs, event_refs = (
        trace.get("formulas"), trace.get("formula_refs"), trace.get("refs"))
    _require(isinstance(formulas, dict) and set(formulas) == FORMULA_KEYS,
             "trace.formulas must contain exactly the documented QKV formulas")
    _require(isinstance(formula_refs, dict) and set(formula_refs) == FORMULA_KEYS,
             "trace.formula_refs must cite every formula")
    _require(isinstance(event_refs, dict) and set(event_refs) == EVENT_REFS,
             "trace.refs must cite every event kind")
    for key in FORMULA_KEYS:
        _evidence(formula_refs[key], refs, f"trace.formulas.{key}")
        _require(isinstance(formulas[key], str) and formulas[key].strip(),
                 f"trace.formulas.{key} must be a nonempty expression")
    for key in EVENT_REFS:
        _evidence(event_refs[key], refs, f"trace.events.{key}")
    scenarios = document.get("scenarios")
    _require(isinstance(scenarios, list) and len(scenarios) == 2 and
             {item.get("id") for item in scenarios if isinstance(item, dict)} ==
             {"aligned", "padded"}, "v2 requires aligned and padded scenarios")
    base["scenarios"] = [_scenario(document, scenario, formulas, formula_refs, event_refs)
                         for scenario in scenarios]
    return base, source_text
