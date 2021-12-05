#include <iostream>
void merge_arrays(int* arr1, int len1, int* arr2, int len2) {
    int j = 0;

    while (j < len2) {
        for (int i = 0; i < len1 + len2; i++) {
            if (arr1[i] >= arr2[j] || i == len1 + j) {
                for (int k = len1 + j; k > i; k--) {
                    arr1[k] = arr1[k-1];
                }
                arr1[i] = arr2[j];
                break;
            }
        }
        j++;
    }
}

