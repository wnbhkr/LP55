#include <iostream>

// CUDA kernel for matrix multiplication
__global__ void matrixMultiplication(int *a, int *b, int *c, int rowsA, int colsA, int colsB) {
    // Calculate global row and column indices for the current thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Perform matrix multiplication if within matrix dimensions
    if (row < rowsA && col < colsB) {
        int sum = 0;
        for (int i = 0; i < colsA; ++i) {
            sum += a[row * colsA + i] * b[i * colsB + col];
        }
        c[row * colsB + col] = sum;
    }
}

int main() {
    const int rowsA = 3;
    const int colsA = 3;
    const int colsB = 3;
    const int sizeA = rowsA * colsA;
    const int sizeB = colsA * colsB;
    const int sizeC = rowsA * colsB;

    int *h_a, *h_b, *h_c; // Host matrices
    int *d_a, *d_b, *d_c; // Device matrices

    // Allocate memory for host matrices
    h_a = new int[sizeA];
    h_b = new int[sizeB];
    h_c = new int[sizeC];

    // Initialize host matrices
    for (int i = 0; i < sizeA; ++i) {
        h_a[i] = i + 1;
    }
    for (int i = 0; i < sizeB; ++i) {
        h_b[i] = i + 1;
    }

    // Allocate memory for device matrices
    cudaMalloc(&d_a, sizeA * sizeof(int));
    cudaMalloc(&d_b, sizeB * sizeof(int));
    cudaMalloc(&d_c, sizeC * sizeof(int));

    // Copy host matrices to device
    cudaMemcpy(d_a, h_a, sizeA * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeB * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    // launches N blocks, and each block contains only one thread.
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((colsB + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rowsA + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch kernel
    matrixMultiplication<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, rowsA, colsA, colsB);

    // Copy result back to host
    cudaMemcpy(h_c, d_c, sizeC * sizeof(int), cudaMemcpyDeviceToHost);

    // Print result
    std::cout << "Result matrix:" << std::endl;
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            std::cout << h_c[i * colsB + j] << " ";
        }
        std::cout << std::endl;
    }

    // Free memory
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
