#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

// Merge two sorted subarrays arr[l..m] and arr[m+1..r]
void merge(vector<int>& arr, int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;

    // Create temporary arrays
    vector<int> L(n1), R(n2);

    // Copy data to temporary arrays L[] and R[]
    for (int i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (int j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    // Merge the temporary arrays back into arr[l..r]
    int i = 0, j = 0, k = l;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    // Copy the remaining elements of L[], if any
    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    // Copy the remaining elements of R[], if any
    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

// Parallel merge sort function
void parallelMergeSort(vector<int>& arr, int l, int r) {
    if (l < r) {
        // Calculate mid point
        int m = l + (r - l) / 2;

        #pragma omp parallel sections
        {
            // Recursively sort first and second halves
            #pragma omp section
            parallelMergeSort(arr, l, m);
            #pragma omp section
            parallelMergeSort(arr, m + 1, r);
        }

        // Merge the sorted halves
        merge(arr, l, m, r);
    }
}

// Function to print an array
void printArray(const vector<int>& arr) {
    for (int num : arr)
        cout << num << " ";
    cout << endl;
}

int main() {
    vector<int> arr = {12, 11, 13, 5, 6, 7};
    int n = arr.size();

    cout << "Original array: ";
    printArray(arr);

    // Perform parallel merge sort
    parallelMergeSort(arr, 0, n - 1);

    cout << "Sorted array: ";
    printArray(arr);

    return 0;
}
