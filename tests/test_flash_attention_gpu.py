"""Optional device check against an independent PyTorch attention reference."""

import importlib.util
from pathlib import Path
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "examples/sources/triton_flash_attention2.py"


class FlashAttentionGpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            import triton  # noqa: F401 - required by the loaded teaching kernel
        except ImportError as exc:
            raise unittest.SkipTest(f"PyTorch and Triton are required: {exc}") from exc
        if not torch.cuda.is_available():
            raise unittest.SkipTest("an NVIDIA CUDA device is required")
        spec = importlib.util.spec_from_file_location("op_show_flash_attention_demo", SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.torch = torch
        cls.kernel = module

    def test_aligned_and_tail_match_full_softmax(self):
        torch = self.torch
        torch.manual_seed(7)
        for m, n in ((32, 32), (27, 21)):
            with self.subTest(m=m, n=n):
                q = torch.randn((m, 16), device="cuda", dtype=torch.float16)
                k = torch.randn((n, 16), device="cuda", dtype=torch.float16)
                v = torch.randn((n, 16), device="cuda", dtype=torch.float16)
                out = torch.empty((m, 16), device="cuda", dtype=torch.float32)
                self.kernel.launch_flash_attention2_forward_demo(q, k, v, out)
                scores = (q.float() @ k.float().T) * (16 ** -0.5)
                expected = torch.softmax(scores, dim=-1) @ v.float()
                torch.testing.assert_close(out, expected, atol=0.03, rtol=0.03)


if __name__ == "__main__":
    unittest.main()
