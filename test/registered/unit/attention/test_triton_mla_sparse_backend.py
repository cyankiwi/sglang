"""CPU-only contract tests for the opt-in DSA adapter (no model or GPU runs).

Run directly with unittest:
    python test/registered/unit/attention/test_triton_mla_sparse_backend.py

Only CUDA tensor residency, index-transform kernels, concatenation, and the new
kernel are mocked. The actual backend dispatch, padding, configuration passes,
and KPool gates run unchanged. Kernel numerical/graph tests live separately.
"""

import argparse
import sys
import types
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, PropertyMock, patch

import torch
from sglang.srt.arg_groups import overrides
from sglang.srt.arg_groups.arg_utils import add_cli_args_from_dataclass
from sglang.srt.arg_groups.fields.exec_ import ExecKernel
from sglang.srt.layers.attention import dsa_backend as dsa
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

BACKEND = "triton_mla_sparse"
KERNEL_MODULE = "sglang.kernels.ops.attention.dsa.triton_mla_sparse"


class TestTritonMLASparseAdapter(unittest.TestCase):
    def setUp(self):
        self.backend = dsa.DeepseekSparseAttnBackend.__new__(
            dsa.DeepseekSparseAttnBackend
        )
        self.backend.device = "cuda:0"
        self.backend.device_sm_major = 12
        self.backend.num_splits = 0
        self.backend._triton_mla_sparse_sm_count = 48
        self.backend.kv_lora_rank = 512
        self.backend.qk_rope_head_dim = 0
        self.backend.kv_cache_dtype = torch.bfloat16
        self.backend.dsa_kv_cache_store_fp8 = False
        self.backend.dsa_prefill_impl = BACKEND
        self.backend.dsa_decode_impl = BACKEND
        self.backend.dsa_index_kpool = 4
        self.backend.use_mha = False
        self.backend.hisparse_coordinator = None
        self.backend.speculative_num_steps = 0
        self.backend.use_fused_topk = True
        self.backend.dsa_topk_backend = NS(is_sgl_kernel=lambda: True)
        self.events = []
        self.kernel = Mock(side_effect=self._kernel)
        module = types.ModuleType(KERNEL_MODULE)
        module.triton_mla_sparse_attention = self.kernel
        self._patch(patch.dict(sys.modules, {KERNEL_MODULE: module}))
        self._patch(patch.object(dsa, "is_cuda", return_value=True))
        self._patch(patch.object(dsa, "_is_hip", False))
        # All storage remains on CPU. Do not replace device or launch CUDA ops.
        self.residency = self._patch(
            patch.object(
                torch.Tensor, "is_cuda", new_callable=PropertyMock, return_value=True
            )
        )
        self.concat = self._patch(
            patch.object(
                dsa,
                "concat_mla_absorb_q_general",
                side_effect=lambda a, b: torch.cat((a, b), -1),
            )
        )
        self.props = self._patch(
            patch.object(
                torch.cuda,
                "get_device_properties",
                side_effect=AssertionError("device query in forward"),
            )
        )

    def _patch(self, context):
        result = context.start()
        self.addCleanup(context.stop)
        return result

    def _kernel(self, **kwargs):
        self.events.append("kernel")
        q = kwargs["q"]
        self.output = q.new_zeros((*q.shape[:2], 512))
        return self.output

    def inputs(self, rope=0, tokens=2, width=2051):
        return {
            "q_all": None,
            "q_nope": torch.zeros(tokens, 4, 512, dtype=torch.bfloat16),
            "q_rope": torch.ones(tokens, 4, rope, dtype=torch.bfloat16),
            "kv_cache": torch.arange(32, dtype=torch.bfloat16)
            .view(32, 1, 1)
            .expand(32, 1, 512 + rope)
            .contiguous(),
            "v_head_dim": 512,
            "page_table_1": torch.full((tokens, width), -1, dtype=torch.int32),
            "sm_scale": 256**-0.5,
        }

    def call(self, **kwargs):
        return self.backend._forward_triton_mla_sparse(**kwargs)

    def test_tail_lengths_zero_through_three_and_full_table_preserved(self):
        for tail in range(4):
            for dtype in (torch.int32, torch.int64):
                with self.subTest(tail=tail, dtype=dtype):
                    inp = self.inputs()
                    indices = inp["page_table_1"].to(dtype)
                    # Different physical slots in each row, only in the KPool tail.
                    indices[0, 2048 : 2048 + tail] = torch.tensor([19, 7, 31][:tail])
                    indices[1, 2048 : 2048 + tail] = torch.tensor([5, 29, 11][:tail])
                    inp["page_table_1"] = indices
                    result = self.call(**inp)
                    actual = self.kernel.call_args.kwargs
                    self.assertIs(result, self.output)
                    self.assertEqual(result.shape, (2, 4, 512))
                    self.assertEqual(result.dtype, torch.bfloat16)
                    self.assertIs(actual["q"], inp["q_nope"])
                    self.assertIs(actual["kv"], inp["kv_cache"])
                    self.assertEqual(actual["indices"].shape, (2, 1, 2064))
                    self.assertEqual(actual["indices"].dtype, dtype)
                    torch.testing.assert_close(actual["indices"][:, 0, :2051], indices)
                    self.assertTrue((actual["indices"][:, :, 2051:] == -1).all())
                    self.assertEqual(actual["sm_scale"], 256**-0.5)
                    self.assertEqual(actual["sm_count"], 48)
                    self.assertIsNone(actual["num_kv_splits"])
        self.concat.assert_not_called()
        self.props.assert_not_called()

    def test_aligned_indices_and_precombined_q_are_reused(self):
        inp = self.inputs(rope=64, width=2064)
        inp["q_all"] = torch.cat((inp["q_nope"], inp["q_rope"]), -1)
        self.call(**inp)
        args = self.kernel.call_args.kwargs
        self.assertIs(args["q"], inp["q_all"])
        self.assertEqual(args["indices"].data_ptr(), inp["page_table_1"].data_ptr())
        self.concat.assert_not_called()

    def test_separate_rope_is_concatenated_once(self):
        inp = self.inputs(rope=64)
        self.call(**inp)
        self.concat.assert_called_once()
        torch.testing.assert_close(
            self.kernel.call_args.kwargs["q"],
            torch.cat((inp["q_nope"], inp["q_rope"]), -1),
        )

    def test_outer_strides_are_preserved_and_index_inner_stride_normalized(self):
        inp = self.inputs(rope=64, width=32)
        q = torch.zeros(4, 4, 576, dtype=torch.bfloat16)[::2]
        kv = torch.zeros(64, 1, 576, dtype=torch.bfloat16)[::2]
        inp.update(q_all=q, kv_cache=kv, page_table_1=inp["page_table_1"][:, ::2])
        self.call(**inp)
        args = self.kernel.call_args.kwargs
        self.assertIs(args["q"], q)
        self.assertIs(args["kv"], kv)
        self.assertEqual(args["indices"].stride(-1), 1)

    def test_deterministic_uses_explicit_one_split(self):
        self.backend.num_splits = 1
        self.call(**self.inputs())
        self.assertEqual(self.kernel.call_args.kwargs["num_kv_splits"], 1)

    def test_rejects_platform(self):
        for major in (9, 10, 11, 13):
            self.backend.device_sm_major = major
            with self.assertRaisesRegex(ValueError, "SM12x"):
                self.call(**self.inputs())
        self.backend.device_sm_major = 12
        with (
            patch.object(dsa, "_is_hip", True),
            self.assertRaisesRegex(ValueError, "SM12x"),
        ):
            self.call(**self.inputs())
        self.residency.return_value = False
        with self.assertRaisesRegex(ValueError, "SM12x"):
            self.call(**self.inputs())
        self.kernel.assert_not_called()

    def test_rejects_dtype_shape_and_device_mismatch(self):
        base = self.inputs(rope=64)
        cases = [
            {"q_nope": base["q_nope"].float()},
            {"q_rope": base["q_rope"].half()},
            {"kv_cache": base["kv_cache"].to(torch.float8_e4m3fn)},
            {"v_head_dim": 256},
            {"q_all": torch.zeros(2, 4, 256, dtype=torch.bfloat16)},
            {"q_all": torch.zeros(2, 512, dtype=torch.bfloat16)},
            {"q_all": torch.tensor(0, dtype=torch.bfloat16)},
            {"q_rope": base["q_rope"][:, :, :32]},
            {"q_rope": base["q_rope"][:1]},
            {"kv_cache": base["kv_cache"].expand(32, 2, 576)},
            {"kv_cache": base["kv_cache"][:, :, :512]},
            {"kv_cache": base["kv_cache"].to("meta")},
            {"q_nope": torch.empty(2, 4, 1024, dtype=torch.bfloat16)[:, :, ::2]},
            {"page_table_1": base["page_table_1"].float()},
            {"page_table_1": base["page_table_1"].unsqueeze(1)},
            {"page_table_1": base["page_table_1"][:1]},
            {"page_table_1": base["page_table_1"].to("meta")},
        ]
        for change in cases:
            with (
                self.subTest(
                    change={k: getattr(v, "shape", v) for k, v in change.items()}
                ),
                self.assertRaises(ValueError),
            ):
                self.call(**(base | change))
        self.kernel.assert_not_called()

    def test_real_prefill_and_decode_dispatch(self):
        for mode in (ForwardMode.EXTEND, ForwardMode.DECODE):
            for fused in (False, True):
                for rope, combined in ((0, False), (0, True), (64, False), (64, True)):
                    with self.subTest(
                        mode=mode, fused=fused, rope=rope, combined=combined
                    ):
                        self.events.clear()
                        self.concat.reset_mock()
                        self.kernel.reset_mock()
                        inp = self.inputs(rope=rope)
                        physical = inp["page_table_1"]
                        physical[:, -3:] = torch.tensor([[19, 7, 31], [5, 29, 11]])
                        pool = NS(
                            get_key_buffer=Mock(return_value=inp["kv_cache"]),
                            set_mla_kv_buffer=Mock(
                                side_effect=lambda *a: self.events.append("cache_write")
                            ),
                        )
                        self.backend.token_to_kv_pool = pool
                        self.backend.use_fused_topk = fused
                        self.backend.forward_metadata = NS(
                            page_table_1=torch.tensor(
                                [[31, 7, 19], [11, 29, 5]], dtype=torch.int32
                            ),
                            dsa_extend_seq_lens_list=[1, 1],
                            cu_seqlens_q=torch.tensor([0, 1, 2], dtype=torch.int32),
                        )
                        batch = NS(
                            forward_mode=mode, out_cache_loc=torch.tensor([19, 5])
                        )
                        layer = NS(
                            is_cross_attention=False,
                            layer_id=3,
                            tp_q_head_num=4,
                            v_head_dim=512,
                            head_dim=512 + rope,
                            scaling=256**-0.5,
                        )
                        # Fused indexer already returns physical rows. Unfused
                        # transform is the sole owner of request->physical mapping.
                        logical = torch.zeros_like(physical)

                        def transform(logical=logical, physical=physical, **kwargs):
                            self.events.append("transform")
                            self.assertIs(
                                kwargs["page_table"],
                                self.backend.forward_metadata.page_table_1,
                            )
                            self.assertIs(kwargs["topk_indices"], logical)
                            self.assertEqual(kwargs["page_size"], 1)
                            return physical

                        name = (
                            "transform_index_page_table_prefill"
                            if mode == ForwardMode.EXTEND
                            else "transform_index_page_table_decode"
                        )
                        q = (
                            torch.cat((inp["q_nope"], inp["q_rope"]), -1)
                            if combined
                            else inp["q_nope"]
                        )
                        forward = (
                            self.backend.forward_extend
                            if mode == ForwardMode.EXTEND
                            else self.backend.forward_decode
                        )
                        with patch.object(dsa, name, side_effect=transform) as mapping:
                            result = forward(
                                q,
                                inp["kv_cache"][:2],
                                inp["kv_cache"][:2],
                                layer,
                                batch,
                                q_rope=None if combined else inp["q_rope"],
                                topk_indices=physical if fused else logical,
                            )
                        self.assertEqual(mapping.call_count, 0 if fused else 1)
                        self.kernel.assert_called_once()
                        self.assertIs(result, self.output)
                        self.assertEqual(
                            self.events,
                            (
                                ["cache_write", "kernel"]
                                if fused
                                else ["cache_write", "transform", "kernel"]
                            ),
                        )
                        torch.testing.assert_close(
                            self.kernel.call_args.kwargs["indices"][:, 0, :2051],
                            physical,
                        )
                        self.assertEqual(
                            self.concat.call_count, int(rope != 0 and not combined)
                        )
                        if combined or rope == 0:
                            self.assertEqual(
                                self.kernel.call_args.kwargs["q"].data_ptr(),
                                q.data_ptr(),
                            )
        self.props.assert_not_called()

    def test_attention_sink_rejected_before_cache_write_or_kernel(self):
        self.backend.forward_metadata = NS()
        layer = NS(is_cross_attention=False)
        for mode, forward in (
            (ForwardMode.EXTEND, self.backend.forward_extend),
            (ForwardMode.DECODE, self.backend.forward_decode),
        ):
            with (
                self.subTest(mode=mode),
                self.assertRaisesRegex(RuntimeError, "attention sinks"),
            ):
                forward(
                    None,
                    None,
                    None,
                    layer,
                    NS(forward_mode=mode),
                    attn_sink=torch.zeros(4),
                )
        self.kernel.assert_not_called()

    def test_padded_graph_rows_remain_masked(self):
        inp = self.inputs(tokens=4)
        live = inp["page_table_1"][:1].clone()
        live[0, -1] = 31
        inp["page_table_1"] = self.backend._pad_topk_indices(live, 4)
        self.call(**inp)
        actual = self.kernel.call_args.kwargs["indices"]
        self.assertEqual(actual[0, 0, 2050], 31)
        self.assertTrue((actual[1:] == -1).all())

    def test_kpool_gate_no_fallback_and_old_triton_still_excluded(self):
        indices = self.inputs()["page_table_1"]
        self.assertEqual(
            self.backend._resolve_kpool_tail_backend(indices, BACKEND), BACKEND
        )
        for phase in ("prefill", "decode"):
            self.backend._check_kpool_tail_backend(indices, BACKEND, phase)
            with self.assertRaises(NotImplementedError):
                self.backend._check_kpool_tail_backend(indices, "triton", phase)
        self.assertEqual(
            self.backend._resolve_kpool_tail_backend(indices, "flashmla_sparse"),
            "trtllm",
        )

    def test_bf16_transform_is_paged(self):
        for mode in (ForwardMode.EXTEND, ForwardMode.DECODE):
            self.assertEqual(
                self.backend.get_topk_transform_method(mode),
                dsa.TopkTransformMethod.PAGED,
            )

    def test_runtime_init_checks_and_static_sm_count(self):
        runner = NS(
            model_config=NS(
                dtype=torch.bfloat16, has_attention_sinks=False, v_head_dim=256
            )
        )
        parallel = NS(dcp_size=1, attn_cp_size=1, enable_prefill_cp=False)
        spec = NS(speculative_algorithm=None)
        disagg = NS(disaggregation_mode="null")
        with (
            patch.object(dsa, "get_parallel", return_value=parallel),
            patch.object(dsa, "get_spec", return_value=spec),
            patch.object(dsa, "get_disagg", return_value=disagg),
        ):
            self.props.side_effect = None
            self.props.return_value = NS(multi_processor_count=48)
            self.backend._init_triton_mla_sparse(runner)
            self.props.assert_called_once_with("cuda:0")
            self.assertEqual(self.backend._triton_mla_sparse_sm_count, 48)
            for obj, attr, value in (
                (self.backend, "kv_lora_rank", 256),
                (self.backend, "qk_rope_head_dim", 32),
                (self.backend, "dsa_kv_cache_store_fp8", True),
                (runner.model_config, "dtype", torch.float16),
                (runner.model_config, "has_attention_sinks", True),
                (self.backend, "hisparse_coordinator", object()),
                (parallel, "dcp_size", 2),
                (parallel, "attn_cp_size", 2),
                (parallel, "enable_prefill_cp", True),
                (spec, "speculative_algorithm", "EAGLE"),
                (self.backend, "speculative_num_steps", 2),
                (disagg, "disaggregation_mode", "decode"),
            ):
                with (
                    self.subTest(attr=attr),
                    patch.object(obj, attr, value),
                    self.assertRaises(ValueError),
                ):
                    self.backend._init_triton_mla_sparse(runner)


class TestTritonMLASparseConfig(unittest.TestCase):
    def setUp(self):
        self.platform = NS(is_cuda=True, is_hip=False, is_npu=False, is_xpu=False)
        self.model = NS(
            dtype=torch.bfloat16,
            kv_lora_rank=512,
            qk_rope_head_dim=0,
            v_head_dim=256,
            has_attention_sinks=False,
            hf_config=NS(
                architectures=["Glm5NextForConditionalGeneration"],
                index_topk=2048,
                index_kpool=4,
            ),
        )
        for context in (
            patch.object(overrides, "get_platform", return_value=self.platform),
            patch.object(overrides, "model_config_of", return_value=self.model),
            patch.object(torch.cuda, "get_device_capability", return_value=(12, 1)),
        ):
            context.start()
            self.addCleanup(context.stop)

    def view(self, **kwargs):
        fields = {
            "dsa_prefill_backend": "tilelang",
            "dsa_decode_backend": BACKEND,
            "kv_cache_dtype": "bfloat16",
            "enable_hisparse": False,
            "dcp_size": 1,
            "attn_cp_size": 1,
            "enable_prefill_cp": False,
            "speculative_algorithm": None,
            "disaggregation_mode": "null",
            "dsa_topk_backend": "sgl-kernel",
        }
        return overrides.ResolvedView(NS(**(fields | kwargs)))

    def test_cli_choices_and_defaults(self):
        parser = argparse.ArgumentParser()
        add_cli_args_from_dataclass(
            parser, ExecKernel, fields=["dsa_prefill_backend", "dsa_decode_backend"]
        )
        empty = parser.parse_args([])
        self.assertIsNone(empty.dsa_prefill_backend)
        self.assertIsNone(empty.dsa_decode_backend)
        args = parser.parse_args(
            ["--dsa-prefill-backend", BACKEND, "--dsa-decode-backend", BACKEND]
        )
        self.assertEqual(args.dsa_prefill_backend, BACKEND)
        self.assertEqual(args.dsa_decode_backend, BACKEND)

    def test_explicit_selection_preserved_for_kpool_and_absorbed_geometry(self):
        for prefill, decode in (
            ("tilelang", BACKEND),
            (BACKEND, BACKEND),
            (BACKEND, "tilelang"),
        ):
            for rope in (0, 64):
                self.model.qk_rope_head_dim = rope
                view = self.view(dsa_prefill_backend=prefill, dsa_decode_backend=decode)
                self.assertEqual(overrides._dsa_kv_cache_dtype_default(view), {})
                self.assertEqual(overrides._dsa_split_backend_resolution(view), {})
        self.assertEqual(
            overrides._dsa_kv_cache_dtype_default(self.view(kv_cache_dtype="bf16")),
            {"kv_cache_dtype": "bfloat16"},
        )

    def test_rejects_explicit_unsupported_configuration_before_fallbacks(self):
        cases = (
            {"kv_cache_dtype": "auto"},
            {"kv_cache_dtype": "fp8_e4m3"},
            {"enable_hisparse": True},
            {"dcp_size": 2},
            {"attn_cp_size": 2},
            {"enable_prefill_cp": True},
            {"speculative_algorithm": "EAGLE"},
            {"disaggregation_mode": "prefill"},
            {"disaggregation_mode": "decode"},
        )
        for change in cases:
            for check in (
                overrides._dsa_kv_cache_dtype_default,
                overrides._dsa_split_backend_resolution,
            ):
                with (
                    self.subTest(change=change, check=check.__name__),
                    self.assertRaisesRegex(ValueError, BACKEND),
                ):
                    check(self.view(**change))
        for attr, value in (
            ("dtype", torch.float16),
            ("kv_lora_rank", 256),
            ("qk_rope_head_dim", 32),
            ("has_attention_sinks", True),
        ):
            with (
                self.subTest(attr=attr),
                patch.object(self.model, attr, value),
                self.assertRaisesRegex(ValueError, BACKEND),
            ):
                overrides._dsa_split_backend_resolution(self.view())
        for attr in ("is_hip", "is_xpu", "is_npu"):
            with (
                patch.object(self.platform, "is_cuda", False),
                patch.object(self.platform, attr, True),
                self.assertRaisesRegex(ValueError, BACKEND),
            ):
                overrides._dsa_split_backend_resolution(self.view())
        with (
            patch.object(torch.cuda, "get_device_capability", return_value=(10, 0)),
            self.assertRaisesRegex(ValueError, "SM12x"),
        ):
            overrides._dsa_split_backend_resolution(self.view())

    def test_non_dsa_model_rejected_without_reading_mla_geometry(self):
        model = NS(hf_config=NS(architectures=["LlamaForCausalLM"]))
        with (
            patch.object(overrides, "model_config_of", return_value=model),
            self.assertRaisesRegex(ValueError, "DSA model"),
        ):
            overrides._dsa_split_backend_resolution(self.view())

    def test_defaults_unchanged_including_hip(self):
        for hip, major, dtype, expected in (
            (False, 12, "bfloat16", ("flashmla_sparse", "trtllm")),
            (False, 12, "fp8_e4m3", ("trtllm", "trtllm")),
            (False, 9, "bfloat16", ("flashmla_sparse", "fa3")),
            (False, 9, "fp8_e4m3", ("flashmla_kv", "flashmla_kv")),
            (True, 9, "bfloat16", ("triton", "triton")),
            (True, 9, "fp8_e4m3", ("triton", "triton")),
        ):
            with (
                self.subTest(hip=hip, major=major, dtype=dtype),
                patch.object(self.platform, "is_hip", hip),
                patch.object(
                    torch.cuda, "get_device_capability", return_value=(major, 0)
                ),
            ):
                view = self.view(
                    dsa_prefill_backend=None,
                    dsa_decode_backend=None,
                    kv_cache_dtype=dtype,
                )
                self.assertEqual(
                    overrides._dsa_split_backend_resolution(view),
                    {
                        "dsa_prefill_backend": expected[0],
                        "dsa_decode_backend": expected[1],
                    },
                )
        self.assertEqual(
            overrides._dsa_kv_cache_dtype_default(
                self.view(
                    dsa_prefill_backend=None,
                    dsa_decode_backend=None,
                    kv_cache_dtype="auto",
                )
            ),
            {"kv_cache_dtype": "fp8_e4m3"},
        )

    def test_old_rocm_triton_constraints_preserved(self):
        overrides._check_dsa_backend_constraints(
            "fp8_e4m3", "triton", "triton", hip=True
        )
        with self.assertRaisesRegex(ValueError, "ROCm/HIP"):
            overrides._check_dsa_backend_constraints(
                "bfloat16", "triton", "triton", hip=False
            )
        with self.assertRaises(ValueError):
            overrides._check_dsa_backend_constraints(
                "fp8_e4m3", "tilelang", "tilelang", hip=False
            )
        with self.assertRaises(ValueError):
            overrides._check_dsa_backend_constraints(
                "bfloat16", BACKEND, BACKEND, hip=True
            )


if __name__ == "__main__":
    unittest.main()
