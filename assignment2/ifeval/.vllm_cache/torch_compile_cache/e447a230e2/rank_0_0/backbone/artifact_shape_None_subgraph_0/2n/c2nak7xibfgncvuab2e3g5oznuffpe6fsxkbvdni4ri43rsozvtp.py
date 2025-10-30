"""
Compile-time auto-tuning block: 

import torch
from torch._dynamo.testing import rand_strided
from torch._dynamo.utils import preserve_rng_state
from torch._inductor.select_algorithm import AlgorithmSelectorCache
from torch._inductor.async_compile import AsyncCompile

async_compile = AsyncCompile()
generate_example_value = AlgorithmSelectorCache.generate_example_value
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu


triton_per_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0 = async_compile.triton('triton_per_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8192, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'out_ptr0': '*fp16', 'out_ptr2': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '532D1FAFA0AA56BF9A34FBEEAD310A34972CEAD219C7EBCBDE361EDC2DBAC635', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr2, xnumel, r0_numel):
    XBLOCK: tl.constexpr = 1
    r0_numel = 576
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    x0 = xindex
    r0_1 = r0_index
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.int64)
    tmp2 = tl.full([R0_BLOCK], 49152, tl.int32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp1 < 0
    tmp5 = tl.where(tmp4, tmp3, tmp1)
    tl.device_assert((0 <= tmp5) & (tmp5 < 49152), "index out of bounds: 0 <= tmp5 < 49152")
    tmp7 = tl.load(in_ptr1 + (r0_1 + 576*tmp5), r0_mask, other=0.0).to(tl.float32)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp8 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [R0_BLOCK])
    tmp12 = tl.where(r0_mask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp14 = 576.0
    tmp15 = (tmp13 / tmp14)
    tmp16 = 1e-05
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.rsqrt(tmp17)
    tmp19 = tmp8 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tmp22 = tmp20 * tmp21
    tl.store(out_ptr0 + (r0_1 + 576*x0), tmp7, r0_mask)
    tl.store(out_ptr2 + (r0_1 + 576*x0), tmp22, r0_mask)
''', device_str='cuda')


triton_poi_fused_cat_1 = async_compile.triton('triton_poi_fused_cat_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*i64', 'in_ptr2': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '532D1FAFA0AA56BF9A34FBEEAD310A34972CEAD219C7EBCBDE361EDC2DBAC635', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 9)
    x2 = xindex // 576
    x4 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (64*x1 + 960*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (x2), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full([XBLOCK], 8192, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tl.device_assert(((0 <= tl.broadcast_to(tmp10, [XBLOCK])) & (tl.broadcast_to(tmp10, [XBLOCK]) < 8192)) | ~(tmp4 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp10, [XBLOCK]) < 8192")
    tmp12 = tl.load(in_ptr2 + (64*tmp10 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp13 = tmp5 * tmp12
    tmp14 = tl.load(in_ptr0 + (32 + 64*x1 + 960*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp15 = tl.load(in_ptr2 + (32 + 64*tmp10 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp16 = tmp14 * tmp15
    tmp17 = tmp13 - tmp16
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp4, tmp17, tmp18)
    tmp20 = tmp0 >= tmp3
    tmp21 = tl.full([1], 64, tl.int64)
    tmp22 = tmp0 < tmp21
    tmp23 = tl.load(in_ptr0 + (32 + 64*x1 + 960*x2 + ((-32) + x0)), tmp20 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp24 = tl.load(in_ptr1 + (x2), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tl.full([XBLOCK], 8192, tl.int32)
    tmp26 = tmp24 + tmp25
    tmp27 = tmp24 < 0
    tmp28 = tl.where(tmp27, tmp26, tmp24)
    tl.device_assert(((0 <= tl.broadcast_to(tmp28, [XBLOCK])) & (tl.broadcast_to(tmp28, [XBLOCK]) < 8192)) | ~(tmp20 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp28, [XBLOCK]) < 8192")
    tmp30 = tl.load(in_ptr2 + (64*tmp28 + ((-32) + x0)), tmp20 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp31 = tmp23 * tmp30
    tmp32 = tl.load(in_ptr0 + (64*x1 + 960*x2 + ((-32) + x0)), tmp20 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp33 = tl.load(in_ptr2 + (32 + 64*tmp28 + ((-32) + x0)), tmp20 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp34 = tmp32 * tmp33
    tmp35 = tmp31 + tmp34
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp20, tmp35, tmp36)
    tmp38 = tl.where(tmp4, tmp19, tmp37)
    tl.store(out_ptr0 + (x4), tmp38, xmask)
''', device_str='cuda')


triton_poi_fused_cat_2 = async_compile.triton('triton_poi_fused_cat_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*i64', 'in_ptr2': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '532D1FAFA0AA56BF9A34FBEEAD310A34972CEAD219C7EBCBDE361EDC2DBAC635', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 3)
    x2 = xindex // 192
    x4 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (576 + 64*x1 + 960*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (x2), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full([XBLOCK], 8192, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tl.device_assert(((0 <= tl.broadcast_to(tmp10, [XBLOCK])) & (tl.broadcast_to(tmp10, [XBLOCK]) < 8192)) | ~(tmp4 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp10, [XBLOCK]) < 8192")
    tmp12 = tl.load(in_ptr2 + (64*tmp10 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp13 = tmp5 * tmp12
    tmp14 = tl.load(in_ptr0 + (608 + 64*x1 + 960*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp15 = tl.load(in_ptr2 + (32 + 64*tmp10 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp16 = tmp14 * tmp15
    tmp17 = tmp13 - tmp16
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp4, tmp17, tmp18)
    tmp20 = tmp0 >= tmp3
    tmp21 = tl.full([1], 64, tl.int64)
    tmp22 = tmp0 < tmp21
    tmp23 = tl.load(in_ptr0 + (608 + 64*x1 + 960*x2 + ((-32) + x0)), tmp20 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp24 = tl.load(in_ptr1 + (x2), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tl.full([XBLOCK], 8192, tl.int32)
    tmp26 = tmp24 + tmp25
    tmp27 = tmp24 < 0
    tmp28 = tl.where(tmp27, tmp26, tmp24)
    tl.device_assert(((0 <= tl.broadcast_to(tmp28, [XBLOCK])) & (tl.broadcast_to(tmp28, [XBLOCK]) < 8192)) | ~(tmp20 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp28, [XBLOCK]) < 8192")
    tmp30 = tl.load(in_ptr2 + (64*tmp28 + ((-32) + x0)), tmp20 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp31 = tmp23 * tmp30
    tmp32 = tl.load(in_ptr0 + (576 + 64*x1 + 960*x2 + ((-32) + x0)), tmp20 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp33 = tl.load(in_ptr2 + (32 + 64*tmp28 + ((-32) + x0)), tmp20 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp34 = tmp32 * tmp33
    tmp35 = tmp31 + tmp34
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp20, tmp35, tmp36)
    tmp38 = tl.where(tmp4, tmp19, tmp37)
    tl.store(out_ptr0 + (x4), tmp38, xmask)
''', device_str='cuda')


triton_poi_fused_view_3 = async_compile.triton('triton_poi_fused_view_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': '532D1FAFA0AA56BF9A34FBEEAD310A34972CEAD219C7EBCBDE361EDC2DBAC635', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_view_3(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')

async_compile.wait(globals())
del async_compile

import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
with torch.cuda._DeviceGuard(0):
    torch.cuda.set_device(0)
    stream0 = get_raw_stream(0)
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
stream0 = get_raw_stream(0)
arg0_1 = generate_example_value((8192,), (1,), 'cuda:0', torch.int32, 0, (8192,))
arg2_1 = generate_example_value((49152, 576), (576, 1), 'cuda:0', torch.float16, 0, (49152, 576))
arg3_1 = generate_example_value((576,), (1,), 'cuda:0', torch.float16, 0, (576,))
buf0 = generate_example_value((8192, 576), (576, 1), 'cuda:0', torch.float16, 0, (8192, 576))
buf2 = generate_example_value((8192, 576), (576, 1), 'cuda:0', torch.float16, 0, (8192, 576))
triton_per_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0.run(arg0_1, arg2_1, arg3_1, buf0, buf2, 8192, 576, stream=stream0)
del arg0_1, arg2_1, arg3_1, buf0, buf2

stream0 = get_raw_stream(0)
buf3 = generate_example_value((8192, 960), (960, 1), 'cuda:0', torch.float16, 0, (8192, 960))
arg5_1 = generate_example_value((8192,), (1,), 'cuda:0', torch.int64, 0, (8192,))
arg6_1 = generate_example_value((8192, 64), (64, 1), 'cuda:0', torch.float16, 0, (8192, 64))
buf4 = generate_example_value((8192, 9, 64), (576, 64, 1), 'cuda:0', torch.float16, 0, (8192, 9, 64))
triton_poi_fused_cat_1.run(buf3, arg5_1, arg6_1, buf4, 4718592, stream=stream0)
del buf4

stream0 = get_raw_stream(0)
buf5 = generate_example_value((8192, 3, 64), (192, 64, 1), 'cuda:0', torch.float16, 0, (8192, 3, 64))
triton_poi_fused_cat_2.run(buf3, arg5_1, arg6_1, buf5, 1572864, stream=stream0)
del buf3, arg5_1, arg6_1, buf5

stream0 = get_raw_stream(0)
buf6 = generate_example_value((8192, 9, 64), (576, 64, 1), 'cuda:0', torch.float16, 0, (8192, 9, 64))
triton_poi_fused_view_3.run(buf6, 4718592, stream=stream0)
del buf6

"""
# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/torchinductor_yxiao986/2t/c2tqi64jqmovdb4f7257yumzxfzyqrsnepcdnj3g6s4tgvonx4vl.py
# Topologically Sorted Source Nodes: [long, embedding, to, pow_1, mean, add, rsqrt, mul, to_1, mul_1], Original ATen: [aten._to_copy, aten.embedding, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add => add_14
#   embedding => embedding
#   long => convert_element_type
#   mean => mean
#   mul => mul_10
#   mul_1 => mul_15
#   pow_1 => pow_1
#   rsqrt => rsqrt
#   to => convert_element_type_1
#   to_1 => convert_element_type_2
# Graph fragment:
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.int64), kwargs = {})
#   %embedding : [num_users=2] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %convert_element_type), kwargs = {})
#   %convert_element_type_1 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_1, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 1e-05), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_14,), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1, %rsqrt), kwargs = {})
#   %convert_element_type_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_10, torch.float16), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_2, %arg3_1), kwargs = {})
triton_per_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0 = async_compile.triton('triton_per_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8192, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'out_ptr0': '*fp16', 'out_ptr2': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '532D1FAFA0AA56BF9A34FBEEAD310A34972CEAD219C7EBCBDE361EDC2DBAC635', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr2, xnumel, r0_numel):
    XBLOCK: tl.constexpr = 1
    r0_numel = 576
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    x0 = xindex
    r0_1 = r0_index
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.int64)
    tmp2 = tl.full([R0_BLOCK], 49152, tl.int32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp1 < 0
    tmp5 = tl.where(tmp4, tmp3, tmp1)
    tl.device_assert((0 <= tmp5) & (tmp5 < 49152), "index out of bounds: 0 <= tmp5 < 49152")
    tmp7 = tl.load(in_ptr1 + (r0_1 + 576*tmp5), r0_mask, other=0.0).to(tl.float32)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp8 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [R0_BLOCK])
    tmp12 = tl.where(r0_mask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp14 = 576.0
    tmp15 = (tmp13 / tmp14)
    tmp16 = 1e-05
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.rsqrt(tmp17)
    tmp19 = tmp8 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tmp22 = tmp20 * tmp21
    tl.store(out_ptr0 + (r0_1 + 576*x0), tmp7, r0_mask)
    tl.store(out_ptr2 + (r0_1 + 576*x0), tmp22, r0_mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yxiao986/ro/croglw36kwp74yipr5thhjy3xsmmm7c6jfenzzrcpd57m4cn6kke.py
# Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat => cat
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%sub_26, %add_97], -1), kwargs = {})
triton_poi_fused_cat_1 = async_compile.triton('triton_poi_fused_cat_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*i64', 'in_ptr2': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '532D1FAFA0AA56BF9A34FBEEAD310A34972CEAD219C7EBCBDE361EDC2DBAC635', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 9)
    x2 = xindex // 576
    x4 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (64*x1 + 960*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (x2), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full([XBLOCK], 8192, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tl.device_assert(((0 <= tl.broadcast_to(tmp10, [XBLOCK])) & (tl.broadcast_to(tmp10, [XBLOCK]) < 8192)) | ~(tmp4 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp10, [XBLOCK]) < 8192")
    tmp12 = tl.load(in_ptr2 + (64*tmp10 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp13 = tmp5 * tmp12
    tmp14 = tl.load(in_ptr0 + (32 + 64*x1 + 960*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp15 = tl.load(in_ptr2 + (32 + 64*tmp10 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp16 = tmp14 * tmp15
    tmp17 = tmp13 - tmp16
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp4, tmp17, tmp18)
    tmp20 = tmp0 >= tmp3
    tmp21 = tl.full([1], 64, tl.int64)
    tmp22 = tmp0 < tmp21
    tmp23 = tl.load(in_ptr0 + (32 + 64*x1 + 960*x2 + ((-32) + x0)), tmp20 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp24 = tl.load(in_ptr1 + (x2), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tl.full([XBLOCK], 8192, tl.int32)
    tmp26 = tmp24 + tmp25
    tmp27 = tmp24 < 0
    tmp28 = tl.where(tmp27, tmp26, tmp24)
    tl.device_assert(((0 <= tl.broadcast_to(tmp28, [XBLOCK])) & (tl.broadcast_to(tmp28, [XBLOCK]) < 8192)) | ~(tmp20 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp28, [XBLOCK]) < 8192")
    tmp30 = tl.load(in_ptr2 + (64*tmp28 + ((-32) + x0)), tmp20 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp31 = tmp23 * tmp30
    tmp32 = tl.load(in_ptr0 + (64*x1 + 960*x2 + ((-32) + x0)), tmp20 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp33 = tl.load(in_ptr2 + (32 + 64*tmp28 + ((-32) + x0)), tmp20 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp34 = tmp32 * tmp33
    tmp35 = tmp31 + tmp34
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp20, tmp35, tmp36)
    tmp38 = tl.where(tmp4, tmp19, tmp37)
    tl.store(out_ptr0 + (x4), tmp38, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yxiao986/cz/cczac2eaclb2qgupewbkhuqbgrjc3yh6kr3ro5drwkli43zgawth.py
# Topologically Sorted Source Nodes: [cat_2], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_2 => cat_1
# Graph fragment:
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%sub_43, %add_159], -1), kwargs = {})
triton_poi_fused_cat_2 = async_compile.triton('triton_poi_fused_cat_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*i64', 'in_ptr2': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '532D1FAFA0AA56BF9A34FBEEAD310A34972CEAD219C7EBCBDE361EDC2DBAC635', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 3)
    x2 = xindex // 192
    x4 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (576 + 64*x1 + 960*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (x2), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full([XBLOCK], 8192, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tl.device_assert(((0 <= tl.broadcast_to(tmp10, [XBLOCK])) & (tl.broadcast_to(tmp10, [XBLOCK]) < 8192)) | ~(tmp4 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp10, [XBLOCK]) < 8192")
    tmp12 = tl.load(in_ptr2 + (64*tmp10 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp13 = tmp5 * tmp12
    tmp14 = tl.load(in_ptr0 + (608 + 64*x1 + 960*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp15 = tl.load(in_ptr2 + (32 + 64*tmp10 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp16 = tmp14 * tmp15
    tmp17 = tmp13 - tmp16
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp4, tmp17, tmp18)
    tmp20 = tmp0 >= tmp3
    tmp21 = tl.full([1], 64, tl.int64)
    tmp22 = tmp0 < tmp21
    tmp23 = tl.load(in_ptr0 + (608 + 64*x1 + 960*x2 + ((-32) + x0)), tmp20 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp24 = tl.load(in_ptr1 + (x2), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tl.full([XBLOCK], 8192, tl.int32)
    tmp26 = tmp24 + tmp25
    tmp27 = tmp24 < 0
    tmp28 = tl.where(tmp27, tmp26, tmp24)
    tl.device_assert(((0 <= tl.broadcast_to(tmp28, [XBLOCK])) & (tl.broadcast_to(tmp28, [XBLOCK]) < 8192)) | ~(tmp20 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp28, [XBLOCK]) < 8192")
    tmp30 = tl.load(in_ptr2 + (64*tmp28 + ((-32) + x0)), tmp20 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp31 = tmp23 * tmp30
    tmp32 = tl.load(in_ptr0 + (576 + 64*x1 + 960*x2 + ((-32) + x0)), tmp20 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp33 = tl.load(in_ptr2 + (32 + 64*tmp28 + ((-32) + x0)), tmp20 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp34 = tmp32 * tmp33
    tmp35 = tmp31 + tmp34
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp20, tmp35, tmp36)
    tmp38 = tl.where(tmp4, tmp19, tmp37)
    tl.store(out_ptr0 + (x4), tmp38, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yxiao986/r4/cr4h3jy34xblqdyqadmceq5d2nr4byxyt7ecdj63qlxv2ag6cqib.py
# Topologically Sorted Source Nodes: [view_3], Original ATen: [aten.view]
# Source node to ATen node mapping:
#   view_3 => full_default
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([%arg1_1, 9, 64], 0.0), kwargs = {dtype: torch.float16, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_view_3 = async_compile.triton('triton_poi_fused_view_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': '532D1FAFA0AA56BF9A34FBEEAD310A34972CEAD219C7EBCBDE361EDC2DBAC635', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_view_3(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1 = args
    args.clear()
    s72 = arg1_1
    assert_size_stride(arg0_1, (s72, ), (1, ))
    assert_size_stride(arg2_1, (49152, 576), (576, 1))
    assert_size_stride(arg3_1, (576, ), (1, ))
    assert_size_stride(arg4_1, (960, 576), (576, 1))
    assert_size_stride(arg5_1, (s72, ), (1, ))
    assert_size_stride(arg6_1, (8192, 64), (64, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((s72, 576), (576, 1), torch.float16)
        buf2 = empty_strided_cuda((s72, 576), (576, 1), torch.float16)
        # Topologically Sorted Source Nodes: [long, embedding, to, pow_1, mean, add, rsqrt, mul, to_1, mul_1], Original ATen: [aten._to_copy, aten.embedding, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0.run(arg0_1, arg2_1, arg3_1, buf0, buf2, s72, 576, stream=stream0)
        del arg0_1
        del arg2_1
        del arg3_1
        buf3 = empty_strided_cuda((s72, 960), (960, 1), torch.float16)
        # Topologically Sorted Source Nodes: [to, pow_1, mean, add, rsqrt, mul, to_1, mul_1, linear], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.mm]
        extern_kernels.mm(buf2, reinterpret_tensor(arg4_1, (576, 960), (1, 576), 0), out=buf3)
        del arg4_1
        buf4 = reinterpret_tensor(buf2, (s72, 9, 64), (576, 64, 1), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        triton_poi_fused_cat_1_xnumel = 576*s72
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf3, arg5_1, arg6_1, buf4, triton_poi_fused_cat_1_xnumel, stream=stream0)
        buf5 = empty_strided_cuda((s72, 3, 64), (192, 64, 1), torch.float16)
        # Topologically Sorted Source Nodes: [cat_2], Original ATen: [aten.cat]
        triton_poi_fused_cat_2_xnumel = 192*s72
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf3, arg5_1, arg6_1, buf5, triton_poi_fused_cat_2_xnumel, stream=stream0)
        del arg5_1
        del arg6_1
        buf6 = empty_strided_cuda((s72, 9, 64), (576, 64, 1), torch.float16)
        # Topologically Sorted Source Nodes: [view_3], Original ATen: [aten.view]
        triton_poi_fused_view_3_xnumel = 576*s72
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3.run(buf6, triton_poi_fused_view_3_xnumel, stream=stream0)
    return (buf4, buf5, reinterpret_tensor(buf3, (s72, 3, 64), (960, 64, 1), 768), buf6, buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.int32)
    arg1_1 = 8192
    arg2_1 = rand_strided((49152, 576), (576, 1), device='cuda:0', dtype=torch.float16)
    arg3_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg4_1 = rand_strided((960, 576), (576, 1), device='cuda:0', dtype=torch.float16)
    arg5_1 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg6_1 = rand_strided((8192, 64), (64, 1), device='cuda:0', dtype=torch.float16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
