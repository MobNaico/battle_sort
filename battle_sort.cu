#include <iostream>
#include <vector>
#include <random>
#include <algorithm> 
#include <omp.h>
#include <cuda_runtime.h>
using namespace std;

#define THREADS_PER_BLOCK 256

// Radix Sort en CPU
void radixSortCPU(vector<int>& arr, int maxElement, int numThreads) {
    int n = arr.size();
    int maxDigits = 0;

    while (maxElement) {
        maxElement /= 10;
        maxDigits++;
    }

    int exp = 1;
    vector<int> output(n);

    for (int i = 0; i < maxDigits; ++i) {
        int count[10] = {0};

        // Paralelización del conteo de dígitos
        #pragma omp parallel for num_threads(numThreads) reduction(+:count[:10])
        for (int j = 0; j < n; j++) {
            int digit = (arr[j] / exp) % 10;
            count[digit]++;
        }

        for (int k = 1; k < 10; k++) {
            count[k] += count[k - 1];
        }

        #pragma omp parallel for num_threads(numThreads)
        for (int j = n - 1; j >= 0; j--) {
            int digit = (arr[j] / exp) % 10;
            output[--count[digit]] = arr[j];
        }

        #pragma omp parallel for num_threads(numThreads)
        for (int j = 0; j < n; j++) {
            arr[j] = output[j];
        }

        exp *= 10;
    }
}

// Merge Sort paralelo en GPU
__global__ void mergeKernel(int* d_arr, int* d_temp, int left, int mid, int right) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int blockStart = left + tid;
    
    if (blockStart <= right) {
        int idx1 = left, idx2 = mid + 1, k = left;
        int localArr[THREADS_PER_BLOCK];  // Usar memoria compartida para subarreglos

        // Copiar parte de los arreglos a la memoria compartida
        if (tid < (mid - left + 1)) localArr[tid] = d_arr[left + tid];
        else localArr[tid] = INT_MAX;

        __syncthreads();  // Asegurarse que todos los hilos carguen sus datos

        // Fusión de los subarreglos
        while (idx1 <= mid && idx2 <= right) {
            if (localArr[tid] <= d_arr[idx2]) {
                d_temp[k++] = localArr[tid++];
            } else {
                d_temp[k++] = d_arr[idx2++];
            }
        }

        // Copiar los elementos restantes
        while (idx1 <= mid) d_temp[k++] = d_arr[idx1++];
        while (idx2 <= right) d_temp[k++] = d_arr[idx2++];
    }
}

__global__ void copyKernel(int* d_arr, int* d_temp, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        d_arr[idx] = d_temp[idx];
    }
}

void mergeSortGPU(vector<int>& arr) {
    int* d_arr, * d_temp;
    int size = arr.size() * sizeof(int);

    cudaMalloc(&d_arr, size);
    cudaMalloc(&d_temp, size);
    cudaMemcpy(d_arr, arr.data(), size, cudaMemcpyHostToDevice);

    int numBlocks = (arr.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Realizar la fusión en etapas
    for (int width = 1; width < arr.size(); width *= 2) {
        for (int left = 0; left < arr.size() - width; left += 2 * width) {
            int mid = left + width - 1;
            int right = min(left + 2 * width - 1, (int)arr.size() - 1);

            mergeKernel<<<numBlocks, THREADS_PER_BLOCK>>>(d_arr, d_temp, left, mid, right);
        }

        // Copiar resultados de vuelta al arreglo original
        copyKernel<<<numBlocks, THREADS_PER_BLOCK>>>(d_arr, d_temp, arr.size());
    }

    cudaMemcpy(arr.data(), d_arr, size, cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
    cudaFree(d_temp);
}

// Programa principal
int main(int argc, char* argv[]) {
    if (argc < 4) {
        cerr << "Uso: ./prog <n> <modo> <nt>\n";
        return 1;
    }

    int n = stoi(argv[1]);
    int mode = stoi(argv[2]);
    int numThreads = stoi(argv[3]);

    vector<int> arr(n);

    // Generar números aleatorios
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dist(0, 1000000);
    for (int i = 0; i < n; i++) {
        arr[i] = dist(gen);
    }

    double start, end;

    if (mode == 0) { // CPU
        int maxElement = *max_element(arr.begin(), arr.end());

        start = omp_get_wtime(); // Timer inicia
        radixSortCPU(arr, maxElement, numThreads);
        end = omp_get_wtime(); // Timer termina

        cout << "Ordenamiento completado en CPU.\n";
    } else if (mode == 1) { // GPU
        start = omp_get_wtime(); // Timer inicia
        mergeSortGPU(arr);
        end = omp_get_wtime(); // Timer termina

        cout << "Ordenamiento completado en GPU.\n";
    } else {
        cerr << "Modo no válido: 0 (CPU), 1 (GPU).\n";
        return 1;
    }

    cout << "Tiempo total de ejecución: " << (end - start) << " segundos.\n";

    return 0;
}
