import json
from pathlib import Path
import tempfile
import unittest

from op_show.expr import ExpressionError, evaluate
from op_show.model import ValidationError, validate_model


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


class ModelTests(unittest.TestCase):
    def test_all_examples_and_tail_masks(self):
        for name in ("cuda_tiled_gemm", "triton_cuda_matmul", "triton_ascend_add"):
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
