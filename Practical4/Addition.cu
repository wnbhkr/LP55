#include <iostream>

// Kernel function to add two large vectors
__global__ void vectorAddition(int *a, int *b, int *c, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    const int size = 1000000; // Size of the vectors
    int *h_a, *h_b, *h_c; // Host vectors
    int *d_a, *d_b, *d_c; // Device vectors

    // Allocate memory for host vectors
    h_a = new int[size];
    h_b = new int[size];
    h_c = new int[size];

    // Initialize host vectors
    for (int i = 0; i < size; ++i) {
        a[i] = i;
        b[i] = size - i;
    }

    // Allocate memory for device vectors
    cudaMalloc(&d_a, size * sizeof(int));
    cudaMalloc(&d_b, size * sizeof(int));
    cudaMalloc(&d_c, size * sizeof(int));

    // Copy host vectors to device
    cudaMemcpy(d_a, h_a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    vectorAddition<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, size);

    // Copy result back to host
    cudaMemcpy(h_c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Verify results
    for (int i = 0; i < 10; ++i) {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;

    // Free host memory
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    return 0;
}
