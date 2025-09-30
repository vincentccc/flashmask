import torch
import pytest
import time
import numpy as np
from flash_attn.flash_attn_interface import flash_attn_func
import flash_attn_2_cuda as flash_attn_cuda

# 确保测试在GPU上运行
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type != 'cuda':
    pytest.skip('This test requires a CUDA device', allow_module_level=True)

# 测试配置参数
@pytest.fixture(params=[
    # (batch_size, seq_len, num_heads, head_dim)
    (1, 384, 16, 128),
    # (2, 2048, 8, 64),
    # (4, 512, 16, 32),
])
def test_config(request):
    return request.param

def generate_inputs(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16):
    """生成测试输入数据"""
    torch.random.manual_seed(42)
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    return q, k, v

def generate_causal_mask(batch_size, seq_len, num_heads):
    """生成因果掩码"""
    # FlashMask的mask格式与标准的不同，需要特殊处理
    # 这里我们使用startend_row_indices_tensor格式来表示因果掩码
    # 对于因果掩码，每个位置i只能关注i及之前的位置
    mask_len = 5
    startend_row_indices = torch.zeros(batch_size, num_heads, seq_len, 2, dtype=torch.int32, device=device)
    for b in range(batch_size):
        for k in range(num_heads):
            for i in range(seq_len):
                # if i < (seq_len - 32 - mask_len):
                #     startend_row_indices[b, k, i, 0] = seq_len
                #     startend_row_indices[b, k, i, 1] = seq_len
                # elif i >= (seq_len - 32 - mask_len) and i < (seq_len - 32):
                #     startend_row_indices[b, k, i, 0] = seq_len - 32
                #     startend_row_indices[b, k, i, 1] = seq_len
                # else:
                    startend_row_indices[b, k, i, 0] = seq_len
                    startend_row_indices[b, k, i, 1] = seq_len
                # startend_row_indices[b, k, i, 0] = seq_len
    return startend_row_indices

# 转换函数
def flashmask_to_densemask(startend_row_indices, dtype, causal=True):
    if startend_row_indices is None:
        return None
    bz, num_head, seq_len, bound_num = startend_row_indices.shape
    m = torch.ones((bz, num_head, seq_len, seq_len), dtype=dtype)
    has_end = (causal and bound_num == 2) or ((not causal) and bound_num == 4)
    for bi in range(bz):
        for hi in range(num_head):
            for j in range(seq_len):
                downstart = startend_row_indices[bi, hi, j, 0]
                if has_end:
                    downend = startend_row_indices[bi, hi, j, 1]
                    m[bi, hi, downstart:downend, j] = 0
                else:
                    m[bi, hi, downstart:, j] = 0
                if causal:
                    m[bi, hi, :j, j] = 0
                else:
                    if has_end:
                        upstart = startend_row_indices[bi, hi, j, 2]
                        upend = startend_row_indices[bi, hi, j, 3]
                        m[bi, hi, upstart:upend, j] = 0
                    else:
                        upend = startend_row_indices[bi, hi, j, 1]
                        m[bi, hi, :upend, j] = 0
    return m

def test_flashmask_fwd_vs_flash_attn_causal(test_config):
    """测试flashmask_fwd与flash_attn_func在因果掩码下的精度和性能对比"""
    batch_size, seq_len, num_heads, head_dim = test_config
    dtype = torch.bfloat16
    dropout = 0.0
    causal = True
    return_softmax = False
    
    # 生成测试数据
    q, k, v = generate_inputs(batch_size, seq_len, num_heads, head_dim, dtype)
    startend_row_indices = generate_causal_mask(batch_size, seq_len, num_heads)
    m = flashmask_to_densemask(startend_row_indices, torch.int32, causal)
    # print(m[0].reshape())
    # 复制输入数据用于两种方法的测试
    q1, k1, v1 = q.clone().transpose(1,2), k.clone().transpose(1,2), v.clone().transpose(1,2)
    q2, k2, v2 = q.clone(), k.clone(), v.clone()
    # 预热GPU
    # _ = flash_attn_func(q1, k1, v1, dropout_p=dropout, causal=causal)
    torch.cuda.synchronize()
    
    # 使用flash_attn_func (标准FlashAttention)
    m_bool = m[0][0].bool().cuda()

    start_time = time.time()
    # out_flash_attn = flash_attn_cuda.flashmask_fwd(
    #     q2, k2, v2, 
    #     None, m, None,
    #     dropout, True, return_softmax, True
    # )
    out_flash_attn = torch.nn.functional.scaled_dot_product_attention(q1, k1, v1, m_bool)
    torch.cuda.synchronize()
    time_flash_attn = time.time() - start_time
    out_flash_attn = out_flash_attn.transpose(1,2)

    # 使用flashmask_fwd (带因果掩码)
    start_time = time.time()
    # try:
    # 调用flashmask_fwd函数
    # 参数: q, k, v, fixed_seed_offset, attn_mask, startend_row_indices, dropout, causal, return_softmax, is_test, rng_name
    # outputs = flash_attn_cuda.flashmask_fwd(
    #     q2, k2, v2, 
    #     startend_row_indices, None, None,
    #     dropout, causal, return_softmax, True
    # )
    outputs = flash_attn_cuda.flashmask_fwd(
        q2, k2, v2, 
        None, m, None,
        dropout, False, return_softmax, True
    )
    torch.cuda.synchronize()
    time_flashmask = time.time() - start_time
    out_flashmask = outputs[0]
    # 检查结果形状是否一致

    assert out_flash_attn.shape == out_flashmask.shape, "Output shapes do not match"

    # 计算绝对误差和相对误差
    abs_error = torch.abs(out_flash_attn - out_flashmask).mean().item()
    abs_max_error = torch.abs(out_flash_attn - out_flashmask).max().item()
    # rel_error = (torch.abs(out_flash_attn - out_flashmask) / (torch.abs(out_flash_attn) + 1e-8)).mean().item()
    
    abs_error_tensor = torch.abs(out_flash_attn - out_flashmask)

    # 找到最大误差值及其位置（返回值为：(最大值, 索引坐标)）
    # max_error_value, max_error_idx = torch.max(abs_error_tensor.view(-1), dim=0)
    # max_error_coords = torch.unravel_index(max_error_idx, abs_error_tensor.shape)
    # print(max_error_value, max_error_idx, max_error_coords)
    print(f"\nTest config: batch={batch_size}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}")
    print(f"FlashAttention time: {time_flash_attn*1000:.2f} ms")
    print(f"FlashMask time: {time_flashmask*1000:.2f} ms")
    print(f"Speedup: {time_flash_attn/time_flashmask:.2f}x")
    print(f"Absolute error: {abs_error:.6f}")
    print(f"Absolute Max error: {abs_max_error:.6f}")
    # print(f"Relative error: {rel_error:.6f}")
    
    # 验证精度是否满足要求 (允许一定的浮点误差)
    rtol = 1e-2  # 相对误差容忍度
    atol = 1e-3  # 绝对误差容忍度
    assert torch.allclose(out_flash_attn, out_flashmask, rtol=rtol, atol=atol), \
        f"Results do not match. Max error: {torch.max(torch.abs(out_flash_attn - out_flashmask))}"
            
    # except Exception as e:
    #     print(f"Error in flashmask_fwd: {e}")
    #     pytest.fail(f"flashmask_fwd execution failed: {e}")

# def test_flashmask_fwd_different_head_dims():
#     """测试不同头维度下的flashmask_fwd功能"""
#     batch_size = 1
#     seq_len = 1024
#     num_heads = 8
    
#     # 测试不同的头维度
#     for head_dim in [32, 64, 128, 256]:
#         q, k, v = generate_inputs(batch_size, seq_len, num_heads, head_dim)
#         startend_row_indices = generate_causal_mask(batch_size, seq_len, num_heads)
        
#         try:
#             if isinstance(flash_attn_cuda, MockFlashAttnCuda):
#                 out_flashmask, _, _, _, _, _, _, _ = flash_attn_cuda.flashmask_fwd(
#                     q, k, v, 
#                     None, None, startend_row_indices, 
#                     0.0, True, False, False, "philox"
#                 )
#             else:
#                 out_flashmask, _, _, _, _, _, _, _ = flash_attn_cuda.flashmask_fwd(
#                     q, k, v, 
#                     None, None, startend_row_indices, 
#                     0.0, True, False, False, "philox"
#                 )
#             torch.cuda.synchronize()
#             assert out_flashmask.shape == q.shape, f"Output shape incorrect for head_dim={head_dim}"
#             print(f"✓ Head dimension {head_dim} passed")
#         except Exception as e:
#             pytest.fail(f"flashmask_fwd failed for head_dim={head_dim}: {e}")

# def test_flashmask_fwd_stability():
#     """测试flashmask_fwd的稳定性（多次运行结果一致性）"""
#     batch_size, seq_len, num_heads, head_dim = (1, 1024, 12, 64)
#     q, k, v = generate_inputs(batch_size, seq_len, num_heads, head_dim)
#     startend_row_indices = generate_causal_mask(batch_size, seq_len, num_heads)
    
#     # 多次运行并检查结果一致性
#     results = []
#     for i in range(3):
#         if isinstance(flash_attn_cuda, MockFlashAttnCuda):
#             out, _, _, _, _, _, _, _ = flash_attn_cuda.flashmask_fwd(
#                 q, k, v, 
#                 None, None, startend_row_indices, 
#                 0.0, True, False, False, "philox"
#             )
#         else:
#             out, _, _, _, _, _, _, _ = flash_attn_cuda.flashmask_fwd(
#                 q, k, v, 
#                 None, None, startend_row_indices, 
#                 0.0, True, False, False, "philox"
#             )
#         results.append(out.clone())
    
#     # 检查所有运行结果是否一致
#     for i in range(1, len(results)):
#         assert torch.allclose(results[0], results[i]), f"Inconsistent results between run 0 and run {i}"
    
#     print("✓ Stability test passed: results are consistent across multiple runs")

if __name__ == "__main__":
    # 如果直接运行此脚本，则执行测试
    test_config = (1, 64, 16, 32)
    test_flashmask_fwd_vs_flash_attn_causal(test_config)
    # test_flashmask_fwd_different_head_dims()
    # test_flashmask_fwd_stability()