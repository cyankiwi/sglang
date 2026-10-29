"""Bounded-memory BF16 sparse MLA tests; runnable directly with unittest.

No models, backend imports, pytest, or runtime autotuning are needed. References
use independent FP32 PyTorch math, not another attention kernel. Peak live test
tensors are well below 1 GiB (the largest cache is under 10 MiB).
"""

import unittest
from unittest import mock

import torch

from sglang.kernels.ops.attention.dsa import triton_mla_sparse as mla
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b", runner_config="1-gpu-small")

SPLITS = (1, 2, 4, 8, 16)
SCALE = 256**-0.5  # Original layer scaling, deliberately not D**-0.5.
HAS_CUDA = (
    torch.cuda.is_available()
    and torch.version.hip is None
    and torch.cuda.get_device_capability()[0] >= 8
)


def reference(q, kv, indices, scale=SCALE):
    """FP32 attention over *valid* physical slots, retaining duplicates."""
    out = torch.zeros((*q.shape[:2], 512), device=q.device, dtype=torch.float32)
    for row in range(q.shape[0]):
        ids = indices[row, 0].long()
        ids = ids[(ids >= 0) & (ids < kv.shape[0])]
        if ids.numel():
            keys = kv[ids, 0].float()
            logits = (q[row].float() @ keys.T) * scale
            out[row] = logits.softmax(-1) @ keys[:, :512]
    return out


class TestSplitPolicy(unittest.TestCase):
    def test_narrow_sm121_policy_and_workload_key(self):
        choose = mla._choose_num_kv_splits
        choose.cache_clear()
        # Prefill must not poison the subsequent decode decision.
        for k in (2048, 2064):
            self.assertEqual(choose(128, 32, 512, k, 48, (12, 1)), 1)
            self.assertEqual(choose(1, 32, 512, k, 48, (12, 1)), 16)
            for t in range(2, 9):
                self.assertEqual(choose(t, 32, 512, k, 48, (12, 1)), 8)
            self.assertEqual(choose(9, 32, 512, k, 48, (12, 1)), 1)
        for args in (
            (8, 32, 576, 2064, 48, (12, 1)),
            (8, 32, 512, 2112, 48, (12, 1)),
            (8, 32, 512, 2048, 80, (12, 1)),
            (8, 32, 512, 2048, 48, (12, 0)),
            (8, 32, 512, 2048, 48, (9, 0)),
            (8, 16, 512, 2048, 48, (12, 1)),
            (8, 64, 512, 2048, 48, (12, 1)),
        ):
            with self.subTest(args=args):
                self.assertNotEqual(choose(*args), 8)
        for t in (0, 1, 8, 128):
            for k in (16, 48, 144, 2064, 4096):
                for sms in (1, 48, 132, 1024):
                    result = choose(t, 17, 576, k, sms, (9, 0))
                    self.assertIn(result, SPLITS)
                    self.assertEqual(k % result, 0)

    def test_cpu_input_rejected(self):
        q = torch.empty(1, 32, 512, dtype=torch.bfloat16)
        kv = torch.empty(1, 1, 512, dtype=torch.bfloat16)
        idx = torch.empty(1, 1, 16, dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "NVIDIA CUDA"):
            mla.triton_mla_sparse_attention(q, kv, idx, SCALE)


@unittest.skipUnless(HAS_CUDA, "requires NVIDIA CUDA with BF16 support")
class TestTritonMLASparse(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7321)
        self.old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False

    def tearDown(self):
        torch.backends.cuda.matmul.allow_tf32 = self.old_tf32

    def inputs(
        self,
        t=4,
        h=32,
        d=512,
        k=2064,
        slots=4096,
        strided=False,
        index_dtype=torch.int32,
    ):
        if strided:
            # Padded and sliced token/head strides, with a storage offset.
            q = torch.randn(t * 2, h * 2, d + 16, device="cuda", dtype=torch.bfloat16)
            q = q[1::2, 1::2, 8 : d + 8]
            kv = torch.randn(slots * 2, 1, d + 16, device="cuda", dtype=torch.bfloat16)
            kv = kv[1::2, :, 8 : d + 8]
            idx = torch.full((t * 2, 1, k + 32), -1, device="cuda", dtype=index_dtype)
            idx = idx[1::2, :, 16 : k + 16]
        else:
            q = torch.randn(t, h, d, device="cuda", dtype=torch.bfloat16)
            kv = torch.randn(slots, 1, d, device="cuda", dtype=torch.bfloat16)
            idx = torch.full((t, 1, k), -1, device="cuda", dtype=index_dtype)
        if k and slots:
            idx.copy_(torch.randint(slots, idx.shape, device="cuda", dtype=index_dtype))
        return q, kv, idx

    def assert_reference(self, output, q, kv, idx, scale=SCALE):
        self.assertEqual(output.shape, (*q.shape[:2], 512))
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(output.device, q.device)
        self.assertTrue(torch.isfinite(output).all().item())
        torch.testing.assert_close(
            output.float(), reference(q, kv, idx, scale), atol=0.025, rtol=0.025
        )

    def test_reference_nope_rope_all_splits_and_strides(self):
        for d in (512, 576):
            for k in (2048, 2064):
                for strided in (False, True):
                    dtype = torch.int64 if strided else torch.int32
                    q, kv, idx = self.inputs(
                        d=d, k=k, strided=strided, index_dtype=dtype
                    )
                    idx[0].fill_(-1)  # Fully empty row.
                    idx[1, :, 7:] = -1  # Empty splits after a short prefix.
                    idx[2, :, :-3] = -1  # Empty splits before a live tail.
                    idx[3, :, ::5] = -1
                    idx[3, 0, 1] = kv.shape[0]  # Positive OOB.
                    idx[3, 0, 2] = torch.iinfo(dtype).max
                    idx[3, 0, 3] = torch.iinfo(dtype).min
                    ref = reference(q, kv, idx)
                    for splits in (*SPLITS, None):
                        with self.subTest(d=d, k=k, strided=strided, splits=splits):
                            got = mla.triton_mla_sparse_attention(
                                q, kv, idx, SCALE, splits
                            )
                            torch.testing.assert_close(
                                got.float(), ref, atol=0.025, rtol=0.025
                            )
                            self.assertEqual(got[0].count_nonzero().item(), 0)
                            self.assertTrue(torch.isfinite(got).all().item())

    def test_generic_multiple16_and_partial_head_tiles(self):
        # K not divisible by BLOCK_N=32 or by split*BLOCK_N exercises the
        # masked final tile in each split (including one-index splits at K16).
        for d in (512, 576):
            for k, h in ((16, 1), (48, 7), (144, 17), (2112, 33), (2080, 32)):
                q, kv, idx = self.inputs(t=2, h=h, d=d, k=k, slots=257)
                idx[1, :, 1:] = -1
                for splits in SPLITS:
                    with self.subTest(d=d, k=k, h=h, splits=splits):
                        got = mla.triton_mla_sparse_attention(q, kv, idx, SCALE, splits)
                        self.assert_reference(got, q, kv, idx)

    def test_automatic_dispatch_and_supplied_sm_count(self):
        sms, capability = mla._device_info(torch.cuda.current_device())
        for t in (1, 8, 128):
            q, kv, idx = self.inputs(t=t, slots=257)
            splits = mla._choose_num_kv_splits(t, 32, 512, 2064, sms, capability)
            forced = mla.triton_mla_sparse_attention(q, kv, idx, SCALE, splits)
            automatic = mla.triton_mla_sparse_attention(q, kv, idx, SCALE)
            supplied = mla.triton_mla_sparse_attention(q, kv, idx, SCALE, sm_count=sms)
            torch.testing.assert_close(automatic, forced, atol=0, rtol=0)
            torch.testing.assert_close(supplied, forced, atol=0, rtol=0)

    def test_original_scaling_and_duplicate_slots(self):
        q, kv, idx = self.inputs(t=2, h=3, k=16, slots=7)
        idx.fill_(-1)
        idx[..., :4] = torch.tensor([2, 2, 0, 1], device="cuda")
        for scale in (0.0, -0.125, SCALE):
            for splits in SPLITS:
                got = mla.triton_mla_sparse_attention(q, kv, idx, scale, splits)
                self.assert_reference(got, q, kv, idx, scale)

    def test_live_kpool_tail_materially_changes_output(self):
        for d in (512, 576):
            q, kv, idx = self.inputs(t=1, d=d)
            q.zero_()
            kv.zero_()
            # Distinct tail values and a very large tail logit: truncating any
            # of the 1..3 live KPool columns gives a measurably different result.
            q[..., 0] = 16
            for tail in range(3):
                kv[3000 + tail, 0, 0] = 20 + tail * 4
                kv[3000 + tail, 0, 1:512] = 2 ** (tail + 1)
            for length in (2048, 2049, 2050, 2051):
                idx.fill_(-1)
                idx[..., :2048] = 0
                for col in range(2048, length):
                    idx[..., col] = 3000 + col - 2048
                for splits in SPLITS:
                    with self.subTest(d=d, length=length, splits=splits):
                        got = mla.triton_mla_sparse_attention(q, kv, idx, SCALE, splits)
                        self.assert_reference(got, q, kv, idx)
                        if length > 2048:
                            self.assertGreater(got[..., 1:].float().mean().item(), 1.5)

    def test_empty_inputs_and_zero_stride_inputs(self):
        for splits in (*SPLITS, None):
            q, kv, idx = self.inputs(t=0)
            with (
                mock.patch.object(
                    mla._sparse_mla_kernel_final, "run", side_effect=AssertionError
                ),
                mock.patch.object(
                    mla._sparse_mla_kernel_split, "run", side_effect=AssertionError
                ),
            ):
                got = mla.triton_mla_sparse_attention(q, kv, idx, SCALE, splits)
            self.assertEqual(got.shape, (0, 32, 512))
            for k, slots in ((0, 17), (16, 0)):
                q, kv, idx = self.inputs(t=2, k=k, slots=slots)
                got = mla.triton_mla_sparse_attention(q, kv, idx, SCALE, splits)
                self.assertEqual(got.count_nonzero().item(), 0)
        q, kv, idx = self.inputs(t=1, h=1, k=48, slots=1)
        q = q.expand(3, 17, 512)
        kv = kv.expand(17, 1, 512)
        idx = idx.expand(3, 1, 48)
        for splits in (1, 16):
            got = mla.triton_mla_sparse_attention(q, kv, idx, SCALE, splits)
            self.assert_reference(got, q, kv, idx)

    def test_negative_logits_below_old_finite_sentinel(self):
        q, kv, idx = self.inputs(t=2, k=2064, slots=2)
        q.zero_()
        q[..., 0] = -1e16
        kv.fill_(1)
        kv[..., 0] = 1e16
        idx.fill_(-1)
        idx[0, 0, -1] = 1
        # The donor's -1e30 sentinel gave padding more weight than this real
        # -6e30 logit. Guarded -inf must retain the sole valid value exactly.
        for splits in SPLITS:
            got = mla.triton_mla_sparse_attention(q, kv, idx, SCALE, splits)
            torch.testing.assert_close(
                got[0], kv[1, 0].expand_as(got[0]), atol=0, rtol=0
            )
            self.assertEqual(got[1].count_nonzero().item(), 0)

    def test_invalid_inputs(self):
        q, kv, idx = self.inputs(t=2, k=16, slots=16)
        cases = [
            ({"q": None}, TypeError, "q must be a torch.Tensor"),
            ({"q": q[0]}, ValueError, "rank-3"),
            ({"kv": kv[0]}, ValueError, "rank-3"),
            ({"indices": idx[:, 0]}, ValueError, "rank-3"),
            ({"q": q.float()}, TypeError, "bfloat16"),
            ({"kv": kv.half()}, TypeError, "bfloat16"),
            ({"indices": idx.float()}, TypeError, "int32 or torch.int64"),
            ({"q": q[..., :256]}, ValueError, "512 or 576"),
            ({"q": q[:, :0]}, ValueError, "H > 0"),
            ({"kv": kv[..., :256]}, ValueError, "kv must have shape"),
            ({"kv": kv.expand(-1, 2, -1)}, ValueError, "kv must have shape"),
            ({"indices": idx[:1]}, ValueError, "matching q"),
            ({"indices": idx.expand(-1, 2, -1)}, ValueError, "matching q"),
            ({"indices": idx[..., :15]}, ValueError, "multiple of 16"),
            ({"q": q[..., ::2]}, ValueError, "stride 1"),
            ({"kv": kv[..., ::2]}, ValueError, "stride 1"),
            ({"indices": idx[..., ::2]}, ValueError, "stride 1"),
            ({"q": q.detach().requires_grad_()}, ValueError, "gradients"),
            ({"kv": kv.detach().requires_grad_()}, ValueError, "gradients"),
            ({"q": q.cpu()}, ValueError, "NVIDIA CUDA"),
            ({"kv": kv.cpu()}, ValueError, "NVIDIA CUDA"),
            ({"indices": idx.cpu()}, ValueError, "NVIDIA CUDA"),
            (
                {"sm_scale": torch.tensor(SCALE, device="cuda")},
                TypeError,
                "host real scalar",
            ),
            ({"sm_scale": True}, TypeError, "host real scalar"),
        ]
        for val in (float("nan"), float("inf"), -float("inf")):
            cases.append(({"sm_scale": val}, ValueError, "finite"))
        for val in (0, -1, 3, 32, 2.0, True, torch.tensor(2)):
            cases.append(({"num_kv_splits": val}, ValueError, "num_kv_splits"))
        for val in (0, -1, 48.0, True, torch.tensor(48)):
            cases.append(({"sm_count": val}, ValueError, "sm_count"))
        for override, error, message in cases:
            with self.subTest(override=list(override), message=message):
                args = dict(q=q, kv=kv, indices=idx, sm_scale=SCALE)
                args.update(override)
                with self.assertRaisesRegex(error, message):
                    mla.triton_mla_sparse_attention(**args)

    @unittest.skipUnless(HAS_CUDA and torch.cuda.device_count() > 1, "needs two GPUs")
    def test_mixed_cuda_devices_rejected(self):
        q, kv, idx = self.inputs(t=1, k=16, slots=16)
        other = (q.device.index + 1) % torch.cuda.device_count()
        for key, tensor in (("kv", kv), ("indices", idx)):
            args = dict(q=q, kv=kv, indices=idx, sm_scale=SCALE)
            args[key] = tensor.to(f"cuda:{other}")
            with self.assertRaisesRegex(ValueError, "same CUDA device"):
                mla.triton_mla_sparse_attention(**args)

    def test_graph_replay_changed_data_empty_splits_and_rows(self):
        for d in (512, 576):
            for splits in (*SPLITS, None):
                with self.subTest(d=d, splits=splits):
                    q, kv, idx = self.inputs(t=8, d=d, slots=257, strided=True)

                    def run():
                        return mla.triton_mla_sparse_attention(
                            q, kv, idx, SCALE, splits
                        )

                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            eager = run()
                    torch.cuda.current_stream().wait_stream(stream)
                    # Warmed capture must not query the device or benchmark.
                    with mock.patch(
                        "torch.cuda.get_device_properties", side_effect=AssertionError
                    ):
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            captured = run()
                    graph.replay()
                    torch.testing.assert_close(captured, eager, atol=0, rtol=0)
                    q.mul_(-0.5)
                    kv.mul_(-1)
                    idx.fill_(-1)
                    idx[0, 0, 2050] = 256  # Sole live KPool tail, rest empty splits.
                    idx[2, 0, :7] = torch.arange(7, device="cuda")
                    graph.replay()
                    self.assert_reference(captured, q, kv, idx)
                    idx.fill_(-1)
                    graph.replay()
                    self.assertEqual(captured.count_nonzero().item(), 0)
                    # Reuse a graph after its all-empty replay, not stale scratch.
                    idx.random_(0, kv.shape[0])
                    graph.replay()
                    self.assert_reference(captured, q, kv, idx)

    def test_sm121_launch_resources(self):
        if torch.cuda.get_device_capability() != (12, 1):
            self.skipTest("SM121 resource qualification")
        compiled = []

        def record(kernel):
            original = kernel.run

            def run(*args, **kwargs):
                result = original(*args, **kwargs)
                compiled.append(result)
                return result

            return mock.patch.object(kernel, "run", side_effect=run)

        with record(mla._sparse_mla_kernel_final), record(mla._sparse_mla_kernel_split):
            for d in (512, 576):
                q, kv, idx = self.inputs(t=1, d=d, slots=257)
                for splits in (1, 16):
                    mla.triton_mla_sparse_attention(q, kv, idx, SCALE, splits)
        limit = torch.cuda.get_device_properties(q.device).shared_memory_per_block_optin
        self.assertEqual(len(compiled), 4)
        for kernel in compiled:
            self.assertLessEqual(kernel.metadata.shared, limit)
            self.assertLessEqual(kernel.n_regs, 255)
            self.assertEqual(kernel.n_spills, 0)


if __name__ == "__main__":
    unittest.main()
