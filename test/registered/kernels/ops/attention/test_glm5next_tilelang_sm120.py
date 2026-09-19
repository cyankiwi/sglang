"""SM120/121 no-RoPE sparse MLA with GLM5Next TP2 shapes.

Run directly with unittest in an environment with TileLang and an SM12 GPU.
This tests the attention kernel, not full-model or speculative correctness.
"""

import unittest

import torch


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12,
    "requires NVIDIA SM120/SM121",
)
class TestGlm5NextTilelangSM120(unittest.TestCase):
    def test_pooled_tail_and_cuda_graph(self):
        from sglang.kernels.ops.attention.dsa.tilelang_kernel import tilelang_sparse_fwd

        torch.manual_seed(123)
        # Pool size 4 adds up to 3 tail tokens to topk=2048. The backend
        # mask-pads the 2051 columns to a multiple of 64 before dispatch.
        lengths = (1, 3, 4, 2047, 2048, 2049, 2050, 2051)
        kv = torch.randn(4096, 1, 512, dtype=torch.bfloat16, device="cuda")
        q = torch.randn(len(lengths), 32, 512, dtype=torch.bfloat16, device="cuda")
        indices = torch.full(
            (len(lengths), 1, 2112), -1, dtype=torch.int32, device="cuda"
        )
        for row, length in enumerate(lengths):
            indices[row, 0, :length] = torch.randperm(4096, device="cuda")[
                :length
            ].int()
        scale = 256**-0.5

        def run():
            return tilelang_sparse_fwd(q, kv, indices, scale).squeeze(0)

        # Compile and warm on a side stream before CUDA graph capture.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                eager = run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, eager, rtol=0, atol=0)
        for row, length in enumerate(lengths):
            with self.subTest(length=length):
                selected = kv[indices[row, 0, :length].long(), 0].float()
                ref = (q[row].float() @ selected.T * scale).softmax(-1) @ selected
                torch.testing.assert_close(
                    captured[row].float(), ref, atol=0.025, rtol=0.025
                )


if __name__ == "__main__":
    unittest.main()
