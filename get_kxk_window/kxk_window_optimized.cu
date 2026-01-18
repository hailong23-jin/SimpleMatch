// kxk_window_optimized.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

// 自动选择线程块大小
inline int get_optimal_block_size(int total_elements, int max_threads=1024) {
    if (total_elements < 256) return 32;
    if (total_elements < 1024) return 64;
    if (total_elements < 4096) return 128;
    if (total_elements < 16384) return 256;
    if (total_elements < 65536) return 512;
    return max_threads;
}

// 前向传播内核
template <typename scalar_t>
__global__ void extract_kxk_window_optimized_kernel(
    const scalar_t* relation_matrix,
    const int64_t* max_indices,
    scalar_t* windows,
    int64_t* top_left_coords,
    int B, int N, int h, int w, int k) {
    
    const int k_half = k / 2;
    const int spatial_size = h * w;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * N * k * k) return;
    
    int b = idx / (N * k * k);
    int remainder = idx % (N * k * k);
    int n = remainder / (k * k);
    remainder = remainder % (k * k);
    int i = remainder / k;
    int j = remainder % k;
    
    int max_idx = max_indices[b * N + n];
    int max_y = max_idx / w;
    int max_x = max_idx % w;
    
    int y_start = max(0, min(max_y - k_half, h - k));
    int x_start = max(0, min(max_x - k_half, w - k));
    
    if (i == 0 && j == 0) {  // 每个窗口的第一个线程写入坐标
        top_left_coords[(b * N + n) * 2] = x_start;
        top_left_coords[(b * N + n) * 2 + 1] = y_start;
    }
    
    int y = y_start + i;
    int x = x_start + j;
    int src_idx = b * N * spatial_size + n * spatial_size + y * w + x;
    windows[idx] = relation_matrix[src_idx];
}

// 反向传播内核
template <typename scalar_t>
__global__ void kxk_window_backward_kernel(
    const scalar_t* grad_windows,
    const int64_t* top_left_coords,
    scalar_t* grad_input,
    int B, int N, int h, int w, int k) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * N * k * k) return;
    
    int b = idx / (N * k * k);
    int remainder = idx % (N * k * k);
    int n = remainder / (k * k);
    remainder = remainder % (k * k);
    int i = remainder / k;
    int j = remainder % k;
    
    int coord_idx = b * N + n;
    int x_start = top_left_coords[coord_idx * 2];
    int y_start = top_left_coords[coord_idx * 2 + 1];
    
    int y = y_start + i;
    int x = x_start + j;
    
    atomicAdd(&grad_input[b * N * h * w + n * h * w + y * w + x], 
              grad_windows[idx]);
}

// 前向传播函数
std::tuple<torch::Tensor, torch::Tensor> extract_kxk_window_optimized(
    torch::Tensor relation_matrix,
    torch::Tensor max_indices,
    int k) {
    
    TORCH_CHECK(relation_matrix.dim() == 3, "relation_matrix must be 3D");
    TORCH_CHECK(max_indices.dim() == 2, "max_indices must be 2D");
    
    int B = relation_matrix.size(0);
    int N = relation_matrix.size(1);
    int hw = relation_matrix.size(2);
    int h = static_cast<int>(std::sqrt(hw));
    int w = h;
    
    auto options = torch::TensorOptions()
        .device(relation_matrix.device())
        .dtype(relation_matrix.dtype());
    auto windows = torch::zeros({B, N, k, k}, options);
    
    auto top_left_coords = torch::zeros({B, N, 2}, 
        torch::TensorOptions()
            .device(relation_matrix.device())
            .dtype(torch::kInt64));
    
    int total_elements = B * N * k * k;
    int threads = get_optimal_block_size(total_elements);
    int blocks = (total_elements + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(relation_matrix.scalar_type(), "extract_kxk_window_optimized", ([&] {
        extract_kxk_window_optimized_kernel<scalar_t><<<blocks, threads>>>(
            relation_matrix.data_ptr<scalar_t>(),
            max_indices.data_ptr<int64_t>(),
            windows.data_ptr<scalar_t>(),
            top_left_coords.data_ptr<int64_t>(),
            B, N, h, w, k);
    }));
    
    return std::make_tuple(windows, top_left_coords);
}

// 反向传播函数
torch::Tensor kxk_window_backward(
    torch::Tensor grad_windows,
    torch::Tensor top_left_coords,
    int B, int N, int h, int w, int k) {
    
    auto grad_input = torch::zeros({B, N, h, w}, 
        grad_windows.options());
    
    int total_elements = B * N * k * k;
    int threads = get_optimal_block_size(total_elements);
    int blocks = (total_elements + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(grad_windows.scalar_type(), "kxk_window_backward", ([&] {
        kxk_window_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_windows.data_ptr<scalar_t>(),
            top_left_coords.data_ptr<int64_t>(),
            grad_input.data_ptr<scalar_t>(),
            B, N, h, w, k);
    }));
    
    return grad_input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("extract_kxk_window_optimized", &extract_kxk_window_optimized, 
          "Optimized parallel extraction of kxk window around max indices");
    m.def("kxk_window_backward", &kxk_window_backward,
          "Backward pass for kxk window extraction");
}