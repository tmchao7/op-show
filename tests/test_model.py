import json
from pathlib import Path
import tempfile
import unittest

from op_show.expr import ExpressionError, evaluate
from op_show.model import ValidationError, validate_model


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


class ModelTests(unittest.TestCase):
    def test_gqa_pid_strategies_keep_worker_and_output_tile_separate(self):
        q_model, _ = validate_model(EXAMPLES / "triton_gqa_q_owned.json")
        kv_model, _ = validate_model(EXAMPLES / "triton_gqa_kv_owned.json")
        for q_case, kv_case in zip(q_model["scenarios"], kv_model["scenarios"]):
            self.assertEqual({tuple(tile["coords"]) for tile in q_case["tiles"]},
                             {tuple(tile["coords"]) for tile in kv_case["tiles"]})
            self.assertEqual(len(q_case["tiles"]), 8)
            self.assertEqual(len(kv_case["tiles"]), 8)
            self.assertEqual(q_case["claim_count"], 1)
            self.assertEqual(kv_case["claim_count"], 4)
            self.assertEqual((q_case["grid"]["x"], q_case["grid"]["y"]), (2, 4))
            self.assertEqual(kv_case["grid"]["x"], 2)
            self.assertEqual([tile["task_group"] for tile in sorted(q_case["tiles"], key=lambda t: t["coords"])],
                             [0, 0, 1, 1, 0, 0, 1, 1])
            for pid in range(2):
                claims = sorted((tile for tile in kv_case["tiles"] if tile["worker"]["x"] == pid),
                                key=lambda tile: tile["claim_index"])
                self.assertEqual([tile["coords"] for tile in claims],
                                 [[0, pid * 2], [1, pid * 2], [0, pid * 2 + 1], [1, pid * 2 + 1]])
                self.assertEqual([tile["start"] for tile in claims], [0, 10, 20, 30])
                self.assertTrue(all(tile["task_group"] == pid for tile in claims))
        self.assertEqual(kv_model["scenarios"][1]["total_ticks"], 40)
        self.assertEqual([step["loop_valid"] for step in kv_model["scenarios"][1]["steps"]
                          if step["id"] == "load-kv"], [16, 5])

    def test_gqa_claim_validation_rejects_missing_or_ambiguous_ownership(self):
        document = json.loads((EXAMPLES / "triton_gqa_kv_owned.json").read_text())
        document["source"]["path"] = str(EXAMPLES / "sources/triton_gqa_pid_strategies.py")
        for mutate, expected in (
            (lambda doc: doc["mapping"]["claim_loop"].update(count="1"), "missing tile coverage"),
            (lambda doc: doc["mapping"]["claim_loop"].update(source_refs=["missing"]),
             "missing source evidence ref"),
            (lambda doc: doc["mapping"]["parallelism"]["task_group"].update(index="H_KV"),
             "task group 2 outside"),
            (lambda doc: doc["mapping"]["parallelism"]["task_group"].update(label_zh=123),
             "label_zh must be a nonempty string"),
            (lambda doc: doc["mapping"]["parallelism"]["task_group"].update(member_axis="N"),
             "member_axis must name an output axis"),
            (lambda doc: doc["mapping"]["tile_axes"][1].update(index="wid_x"),
             "multiple workers"),
        ):
            candidate = json.loads(json.dumps(document))
            mutate(candidate)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "model.json"
                path.write_text(json.dumps(candidate))
                with self.assertRaisesRegex(ValidationError, expected):
                    validate_model(path)

    def test_flash_attention2_query_and_key_tail_slices(self):
        model, source = validate_model(EXAMPLES / "triton_flash_attention2.json")
        self.assertIn("score = tl.dot(q, tl.trans(k))", source)
        self.assertIn("acc = tl.dot(p.to(tl.float16), v, acc)", source)
        aligned, padded = model["scenarios"]
        self.assertEqual(len(aligned["tiles"]), 2)
        self.assertTrue(all(tile["valid_cells"] == 256 for tile in aligned["tiles"]))
        self.assertEqual([step["loop_valid"] for step in padded["steps"] if step["id"] == "load-kv"],
                         [16, 5])
        tail = next(tile for tile in padded["tiles"] if tile["coords"] == [1, 0])
        self.assertEqual(tail["valid_cells"], 11 * 16)
        self.assertEqual([address["samples"][1]["offset"] for address in tail["addresses"]],
                         [16 * 16, 16 * 16, 16 * 16, 16 * 16])
        self.assertEqual((padded["attention_view"]["n_extent"], padded["attention_view"]["n_size"]),
                         (21, 16))

    def test_attention_view_rejects_missing_evidence_and_wrong_key_loop(self):
        document = json.loads((EXAMPLES / "triton_flash_attention2.json").read_text())
        document["source"]["path"] = str(EXAMPLES / "sources/triton_flash_attention2.py")
        for mutate, expected in (
            (lambda doc: doc["mapping"]["attention_view"]["source_refs"].update(score="missing"),
             "missing source evidence ref"),
            (lambda doc: doc["mapping"]["attention_view"].update(n_size="8"),
             "N slices do not match"),
            (lambda doc: doc["mapping"]["attention_view"].update(q_axis="N"),
             "axes must match"),
        ):
            candidate = json.loads(json.dumps(document))
            mutate(candidate)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "model.json"
                path.write_text(json.dumps(candidate))
                with self.assertRaisesRegex(ValidationError, expected):
                    validate_model(path)

    def test_warp_row_reduce_and_gemv_cover_tail_rows_and_k(self):
        for name, input_count in (("cuda_warp_reduce", 1), ("cuda_warp_gemv", 2)):
            with self.subTest(name=name):
                model, source = validate_model(EXAMPLES / f"{name}.json")
                self.assertIn("__shfl_down_sync(0xffffffff, sum, delta)", source)
                aligned, padded = model["scenarios"]
                self.assertEqual([tile["valid_cells"] for tile in aligned["tiles"]], [4, 4])
                self.assertEqual([tile["valid_cells"] for tile in padded["tiles"]], [4, 2])
                self.assertEqual([step["loop_valid"] for step in padded["steps"]
                                  if step["id"] == "accumulate"], [32, 5])
                tail = next(tile for tile in padded["tiles"] if tile["coords"] == [1])
                self.assertEqual(tail["worker"]["x"], 1)
                self.assertEqual((tail["bounds"][0]["start"], tail["bounds"][0]["end"]), (4, 6))
                self.assertEqual([address["samples"][1]["offset"] for address in tail["addresses"]],
                                 ([4 * 37 + 32, 4] if input_count == 1 else [4 * 37 + 32, 32, 4]))
                self.assertEqual(padded["row_k_view"]["kind"], "reduce" if input_count == 1 else "gemv")
                self.assertEqual((padded["row_k_view"]["k_extent"], padded["row_k_view"]["k_size"]),
                                 (37, 32))

    def test_row_k_view_rejects_unproven_input_and_wrong_loop(self):
        document = json.loads((EXAMPLES / "cuda_warp_gemv.json").read_text())
        document["source"]["path"] = str(EXAMPLES / "sources/cuda_warp_gemv.cu")
        for mutate, expected in (
            (lambda doc: doc["mapping"]["row_k_view"]["source_refs"].update(vector="missing"),
             "missing source evidence ref"),
            (lambda doc: doc["mapping"]["row_k_view"].update(k_size="16"),
             "K slices do not match"),
            (lambda doc: doc["mapping"]["row_k_view"].update(output_axis="K"),
             "output axis must match"),
        ):
            candidate = json.loads(json.dumps(document))
            mutate(candidate)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "model.json"
                path.write_text(json.dumps(candidate))
                with self.assertRaisesRegex(ValidationError, expected):
                    validate_model(path)

    def test_all_examples_and_tail_masks(self):
        for name in ("cuda_tiled_gemm", "triton_cuda_matmul", "triton_ascend_add",
                     "cuda_warp_reduce", "cuda_warp_gemv"):
            with self.subTest(name=name):
                model, source = validate_model(EXAMPLES / f"{name}.json")
                self.assertIn(model["kernel"], source)
                aligned, padded = model["scenarios"]
                self.assertEqual(aligned["id"], "aligned")
                self.assertEqual(padded["id"], "padded")
                self.assertTrue(all(t["valid_cells"] == t["total_cells"] for t in aligned["tiles"]))
                self.assertTrue(any(t["valid_cells"] < t["total_cells"] for t in padded["tiles"]))
                self.assertEqual(len({tuple(t["coords"]) for t in padded["tiles"]}), len(padded["tiles"]))
                self.assertEqual(padded["total_ticks"], len(padded["steps"]))

    def test_reduction_tail_is_visible(self):
        for name in ("cuda_tiled_gemm", "triton_cuda_matmul"):
            model, _ = validate_model(EXAMPLES / f"{name}.json")
            scenario = model["scenarios"][1]
            k_steps = [step for step in scenario["steps"] if step["loop_axis"] == "K"]
            self.assertLess(k_steps[-1]["loop_valid"], scenario["params"]["BK"])

    def test_tensor_core_gemm_tiles_and_matrix_slices(self):
        model, source = validate_model(EXAMPLES / "cuda_tensor_core_gemm.json")
        self.assertIn("wmma::mma_sync(acc, a_frag, b_frag, acc)", source)
        self.assertIn("__align__(32) half a_shared", source)
        aligned, padded = model["scenarios"]
        self.assertEqual(len(aligned["tiles"]), 4)
        self.assertTrue(all(tile["valid_cells"] == 256 for tile in aligned["tiles"]))
        self.assertEqual((padded["gemm_view"]["k_extent"], padded["gemm_view"]["k_size"]), (19, 16))
        self.assertEqual([step["loop_valid"] for step in padded["steps"] if step["id"] == "mma"], [16, 3])
        tile = next(tile for tile in padded["tiles"] if tile["coords"] == [1, 1])
        self.assertEqual(tile["valid_cells"], 11 * 5)
        self.assertEqual(tile["bounds"][0]["end"], 27)
        self.assertEqual(tile["bounds"][1]["end"], 21)
        self.assertEqual([address["samples"][1]["offset"] for address in tile["addresses"]],
                         [16 * 19 + 16, 16 * 21 + 16, 16 * 21 + 16])

    def test_gemm_view_rejects_wrong_loop_or_source_ref(self):
        document = json.loads((EXAMPLES / "cuda_tensor_core_gemm.json").read_text())
        document["source"]["path"] = str(EXAMPLES / "sources/cuda_tensor_core_gemm.cu")
        def invalid(mutate, message):
            candidate = json.loads(json.dumps(document))
            mutate(candidate)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "model.json"
                path.write_text(json.dumps(candidate))
                with self.assertRaisesRegex(ValidationError, message):
                    validate_model(path)
        invalid(lambda doc: doc["mapping"]["gemm_view"].update(k_size="8"), "K slices do not match")
        invalid(lambda doc: doc["mapping"]["gemm_view"]["source_refs"].update(mma="missing"),
                "missing source evidence ref")

    def test_optional_chinese_labels_reach_viewer_model(self):
        model, _ = validate_model(EXAMPLES / "cuda_tiled_gemm.json")
        self.assertEqual(model["title_zh"], "CUDA C++ · 分块矩阵乘法")
        self.assertEqual(model["scenarios"][0]["label_zh"], "整除 · 无尾 tile")
        self.assertEqual(model["scenarios"][0]["steps"][0]["label_zh"], "加载 A/B tile")
        self.assertEqual(model["scenarios"][0]["tiles"][0]["addresses"][0]["label_zh"], "A 首元素偏移")
        self._invalid(lambda doc: doc.update(title_zh=""), "model.title_zh must be a nonempty string")

    def test_expression_language_rejects_python_execution(self):
        self.assertEqual(evaluate("ceildiv(N, BN) + wid_x", {"N": 9, "BN": 4, "wid_x": 2}), 5)
        for expression in ("__import__('os').system('echo bad')", "2 ** 20", "N.real", "N[0]"):
            with self.subTest(expression=expression), self.assertRaises(ExpressionError):
                evaluate(expression, {"N": 3})

    def _invalid(self, mutate, expected):
        document = json.loads((EXAMPLES / "cuda_tiled_gemm.json").read_text())
        document["source"]["path"] = str(EXAMPLES / "sources/cuda_tiled_gemm.cu")
        mutate(document)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValidationError, expected):
                validate_model(path)

    def test_rejects_duplicate_and_missing_tiles(self):
        self._invalid(lambda doc: doc["mapping"]["tile_axes"][1].update(index="0"), "multiple workers")
        self._invalid(lambda doc: doc["mapping"]["worker_grid"].update(x="1"), "missing tile coverage")

    def test_rejects_bad_source_ref_and_formula(self):
        self._invalid(lambda doc: doc["source"]["refs"]["grid"].update(contains="not in source"), "does not match")
        self._invalid(lambda doc: doc["mapping"]["tile_axes"][0].update(index="wid_y ** 2"), "unsupported formula")

    def test_rejects_missing_shape_and_dependency_cycle(self):
        self._invalid(lambda doc: doc["scenarios"][0]["params"].pop("M"), "unknown variable")
        def cycle(doc):
            doc["mapping"]["dependencies"] = [
                {"tile": [0, 0], "on": [[0, 1]]},
                {"tile": [0, 1], "on": [[0, 0]]},
            ]
        self._invalid(cycle, "cycle")

    def test_explicit_dependency_delays_only_target_tile(self):
        document = json.loads((EXAMPLES / "cuda_tiled_gemm.json").read_text())
        document["source"]["path"] = str(EXAMPLES / "sources/cuda_tiled_gemm.cu")
        document["mapping"]["dependencies"] = [{"tile": [1, 1], "on": [[0, 0]]}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            path.write_text(json.dumps(document))
            model, _ = validate_model(path)
            for scenario in model["scenarios"]:
                tiles = {tuple(tile["coords"]): tile for tile in scenario["tiles"]}
                self.assertEqual(tiles[(1, 1)]["start"], len(scenario["steps"]))
                self.assertEqual(tiles[(0, 1)]["start"], 0)
                self.assertEqual(scenario["total_ticks"], 2 * len(scenario["steps"]))


    def test_malformed_fields_raise_validation_error(self):
        cases = [
            (lambda doc: doc.update(schema_version=True), "schema_version"),
            (lambda doc: doc.update(schema_version=1.0), "schema_version"),
            (lambda doc: doc.update(backend=[]), "backend must be one of"),
            (lambda doc: doc["mapping"]["tile_axes"][0].update(name={}), "tile axis names"),
            (lambda doc: doc["scenarios"][0].update(id=[1]), "scenario ids"),
        ]
        for mutate, message in cases:
            with self.subTest(message=message):
                self._invalid(mutate, message)

    def test_expression_limits(self):
        from op_show.expr import MAX_EXPRESSION_LENGTH
        with self.assertRaises(ExpressionError):
            evaluate("1" + " + 1" * (MAX_EXPRESSION_LENGTH // 2), {})
        self._invalid(lambda doc: doc["mapping"]["tile_axes"][0].update(
            index="1" + "+1" * 1200), "formula exceeds")

    def test_invalid_utf8_raises_validation_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_bytes(b'{"schema_version": 1, "title": "\xff"}')
            with self.assertRaises(ValidationError):
                validate_model(path)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "bad.cu"
            source.write_bytes(b"// tiled_gemm\n\xff\n")
            document = json.loads((EXAMPLES / "cuda_tiled_gemm.json").read_text())
            document["source"]["path"] = str(source)
            path = directory / "model.json"
            path.write_text(json.dumps(document))
            with self.assertRaises(ValidationError):
                validate_model(path)

    def test_loop_coverage_and_budget_limits(self):
        self._invalid(lambda doc: doc["mapping"]["program"][0].update(repeat="1"),
                      "repeat must equal ceildiv")
        def huge_cells(doc):
            for scenario in doc["scenarios"]:
                scenario["params"].update(M=2000, N=2000, BM=2000, BN=2000, K=1, BK=1)
        self._invalid(huge_cells, "cell budget")
        def address_budget(doc):
            doc["mapping"]["program"] = [{
                "repeat": "ceildiv(K, BK)", "loop_axis": "K", "loop_tile": "BK",
                "loop_source_ref": "loop",
                "steps": [{"id": "load", "label": "Load", "source_ref": "load"}],
            }]
            doc["mapping"]["addresses"] = [{
                "label": "A", "expr": "row * K + k",
                "sample": {"row": "tile_M * BM", "k": "phase * BK"},
                "source_ref": "load",
            }]
            for scenario in doc["scenarios"]:
                params = scenario["params"]
                scenario["id"] = "aligned" if scenario["id"] == "aligned" else "padded"
                params.update(M=256, N=1, BM=1, BN=1, K=96, BK=1)
                if scenario["id"] == "padded":
                    params["M"] = 255
        self._invalid(address_budget, "address sample budget")

    def test_size_ref_is_validated(self):
        document = json.loads((EXAMPLES / "cuda_tiled_gemm.json").read_text())
        document["source"]["path"] = str(EXAMPLES / "sources/cuda_tiled_gemm.cu")
        document["source"]["refs"]["tile_const"] = {"line": 4, "contains": "constexpr int BM = 4"}
        document["mapping"]["tile_axes"][0]["size_ref"] = "tile_const"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            path.write_text(json.dumps(document))
            model, _ = validate_model(path)
            self.assertEqual(model["scenarios"][0]["dimensions"][0]["size_ref"], "tile_const")
        document["mapping"]["tile_axes"][0]["size_ref"] = "missing"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValidationError, "size_ref"):
                validate_model(path)

    def test_single_k_slice_gemm_view_is_valid(self):
        document = json.loads((EXAMPLES / "cuda_tensor_core_gemm.json").read_text())
        document["source"]["path"] = str(EXAMPLES / "sources/cuda_tensor_core_gemm.cu")
        group = document["mapping"]["program"][0]
        for key in ("repeat", "loop_axis", "loop_tile", "loop_source_ref"):
            group.pop(key, None)
        for scenario in document["scenarios"]:
            scenario["params"]["K"] = 16
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            path.write_text(json.dumps(document))
            model, _ = validate_model(path)
            steps = model["scenarios"][0]["steps"]
            self.assertTrue(all(step["iteration"] is None for step in steps))
            self.assertEqual(model["scenarios"][0]["gemm_view"]["k_size"], 16)

    def test_unknown_mapping_has_no_playable_scenarios(self):
        document = json.loads((EXAMPLES / "cuda_tiled_gemm.json").read_text())
        document["source"]["path"] = str(EXAMPLES / "sources/cuda_tiled_gemm.cu")
        document["mapping"] = {"status": "unknown", "note": "Dynamic pointer mapping needs review"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            path.write_text(json.dumps(document))
            model, _ = validate_model(path)
            self.assertEqual(model["scenarios"], [])


if __name__ == "__main__":
    unittest.main()
