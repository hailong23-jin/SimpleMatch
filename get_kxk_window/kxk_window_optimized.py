# kxk_window_optimized.py
import torch
from torch.autograd import Function
import kxk_window_cuda_optimized

class KxKWindowOptimizedFunction(Function):
    @staticmethod
    def forward(ctx, relation_matrix, k):
        B, N, h, w = relation_matrix.shape
        relation_matrix_flat = relation_matrix.reshape(B, N, -1)
        
        max_indices = torch.argmax(relation_matrix_flat, dim=-1)
        
        # 调用优化后的CUDA实现
        windows, top_left_coords = kxk_window_cuda_optimized.extract_kxk_window_optimized(
            relation_matrix_flat, max_indices, k)
        
        ctx.save_for_backward(relation_matrix, top_left_coords)
        ctx.B, ctx.N, ctx.h, ctx.w, ctx.k = B, N, h, w, k
        
        return windows, top_left_coords
    
    @staticmethod
    def backward(ctx, grad_windows, grad_coords):
        relation_matrix, top_left_coords = ctx.saved_tensors
        B, N, h, w, k = ctx.B, ctx.N, ctx.h, ctx.w, ctx.k
        
        # 调用优化后的CUDA反向传播
        grad_input = kxk_window_cuda_optimized.kxk_window_backward(
            grad_windows.contiguous(),
            top_left_coords,
            B, N, h, w, k)
        
        return grad_input.reshape(B, N, h, w), None

def get_kxk_window_optimized(relation_matrix, k):
    """
    全面优化的kxk窗口提取
    
    参数:
        relation_matrix: (B, N, h, w) 输入关系矩阵
        k: 窗口大小
        
    返回:
        windows: (B, N, k, k) 提取的窗口
        top_left_coords: (B, N, 2) 每个窗口的左上角坐标(x,y)
    """
    return KxKWindowOptimizedFunction.apply(relation_matrix, k)