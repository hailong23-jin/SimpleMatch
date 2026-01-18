import torch
from kxk_window_optimized import get_kxk_window_optimized

# 测试数据
B, N, h, w = 8, 3968, 60, 60
k = 30
device = torch.device('cuda')

# 创建需要梯度的输入
relation_matrix = torch.randn(B, N, h, w, device=device, requires_grad=True)

# 提取窗口
import time

start = time.time()
windows, coords = get_kxk_window_optimized(relation_matrix, k)
print(time.time() - start)

# 模拟损失计算
loss = windows.sum()
loss.backward()

print("优化实现完成，梯度计算正确:", relation_matrix.grad is not None)
print("梯度形状:", relation_matrix.grad.shape)