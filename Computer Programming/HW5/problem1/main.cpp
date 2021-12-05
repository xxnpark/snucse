#include <iostream>
#include <iterator>
#include <vector>
#include "header.h"

template <typename T>
void check_array(const T* a, const std::vector<T> b, int N) {
    if (!a || b.size() != N) {
        std::cout << "Failed" << std::endl;
        return;
    }
    for (std::size_t i = 0; i < N; ++i) {
        if (a[i] != b.at(i)) {
            std::cout << "Failed" << std::endl;
            return;
        }
    }
    std::cout << "Passed" << std::endl;
}

template <typename T>
void check_value(const T a, const T b) {
    if (a == b) {
        std::cout << "Passed" << std::endl;
    } else {
        std::cout << "Failed" << std::endl;
    }
}

int main() {
    // Test more with your own test cases

    // 1.1
    std::cout << "1-1" << std::endl;
    check_value(is_palindrome("aabb"), false);
    check_value(is_palindrome(""), true);
    check_value(is_palindrome("jDmvFiNQYGGYQNiFvmDj"), true);
    check_value(is_palindrome("oQhuflCZlHHlZClfuhQo"), true);
    check_value(is_palindrome("jUiuEYXeybbyeXYEuiUj"), true);
    check_value(is_palindrome("sfBrIWljLMMLjlWIrBfs"), true);
    check_value(is_palindrome("erJMzexhmjjmhxezMJre"), true);
    check_value(is_palindrome("RecniiSnllllnSiinceR"), true);
    check_value(is_palindrome("msAEAPKhJooJhKPAEAsm"), true);
    check_value(is_palindrome("JNNHDRGQzyyzQGRDHNNJ"), true);
    check_value(is_palindrome("xHtiKJSoaUUaoSJKitHx"), true);
    check_value(is_palindrome("tDLNjqDWxmmxWDqjNLDt"), true);
    check_value(is_palindrome("CDXjChCajssvsChCuXDC"), false);
    check_value(is_palindrome("troyvruOiOliOrrvylrt"), false);
    check_value(is_palindrome("QAduByaPIyGIPaSBugAQ"), false);
    check_value(is_palindrome("THshhbuyrdsryubhhTHT"), false);
    check_value(is_palindrome("iuPBYXWUBSSnUxXYBPqi"), false);
    check_value(is_palindrome("EZUQrUhMzuujgLUrQUZE"), false);
    check_value(is_palindrome("PohiLKZALerLAZKLZhfP"), false);
    check_value(is_palindrome("FsbmtMMnTHHTnMskmbgF"), false);
    check_value(is_palindrome("XAvQrZudgwauduZrQvAv"), false);
    check_value(is_palindrome("ZqnwtfBlEjTElBptwTqZ"), false);


    // 1.2
    std::cout << "1-2" << std::endl;
    check_value(hamming_distance(1,4), 2);
    check_value(hamming_distance(2147483647,2147483647), 0 );
    check_value(hamming_distance(2147483647,0), 31 );
    check_value(hamming_distance(0,2147483647), 31 );
    check_value(hamming_distance(0,0), 0 );
    check_value(hamming_distance(3,1), 1 );
    check_value(hamming_distance(33,15), 4 );

    // 1.3
    std::cout << "1-3" << std::endl;
    int arr1[]   ={1,3,5,0,0};
    int arr2[] = {3,5};
    merge_arrays(arr1, 3, arr2, 2);
    check_array(arr1, {1,3,3,5,5}, sizeof(arr1)/sizeof(int));

    int arr1_1[]={-6, -3, -2, 0, 2, 3, 0, 0, 0, 0}, arr1_2[]={-7, -6, 6, 7};
    merge_arrays(arr1_1, 6, arr1_2, 4);
    check_array(arr1_1, {-7, -6, -6, -3, -2, 0, 2, 3, 6, 7}, sizeof(arr1_1)/sizeof(int));
    int arr2_1[]={4, 4, 6, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, arr2_2[]={-7, -7, -5, -5, -5, -4, -2, -2, -1, -1, 3};
    merge_arrays(arr2_1, 4, arr2_2, 11);
    check_array(arr2_1, {-7, -7, -5, -5, -5, -4, -2, -2, -1, -1, 3, 4, 4, 6, 6}, sizeof(arr2_1)/sizeof(int));
    int arr3_1[]={-8, -6, -5, -4, -2, -2, 1, 1, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, arr3_2[]={-8, -6, -6, -5, -5, -5, -4, -3, -3, -3, -1, -1, 0, 0, 1, 3};
    merge_arrays(arr3_1, 10, arr3_2, 16);
    check_array(arr3_1, {-8, -8, -6, -6, -6, -5, -5, -5, -5, -4, -4, -3, -3, -3, -2, -2, -1, -1, 0, 0, 1, 1, 1, 2, 3, 5}, sizeof(arr3_1)/sizeof(int));
    int arr4_1[]={-7, -6, -4, -4, -3, -3, -2, -2, -1, 1, 2, 2, 2, 3, 4, 4, 6, 7, 7, 7, 0}, arr4_2[]={4};
    merge_arrays(arr4_1, 20, arr4_2, 1);
    check_array(arr4_1, {-7, -6, -4, -4, -3, -3, -2, -2, -1, 1, 2, 2, 2, 3, 4, 4, 4, 6, 7, 7, 7}, sizeof(arr4_1)/sizeof(int));
    int arr5_1[]={-7, -7, -7, -5, -4, -2, -2, -1, 2, 4, 5, 5, 6, 6, 6, 7, 0, 0}, arr5_2[]={-7, 2};
    merge_arrays(arr5_1, 16, arr5_2, 2);
    check_array(arr5_1, {-7, -7, -7, -7, -5, -4, -2, -2, -1, 2, 2, 4, 5, 5, 6, 6, 6, 7}, sizeof(arr5_1)/sizeof(int));
    int arr6_1[]={3, 0, 0, 0}, arr6_2[]={-2, 0, 3};
    merge_arrays(arr6_1, 1, arr6_2, 3);
    check_array(arr6_1, {-2, 0, 3, 3}, sizeof(arr6_1)/sizeof(int));
    int arr7_1[]={-7, 0, 1, 0}, arr7_2[]={-3};
    merge_arrays(arr7_1, 3, arr7_2, 1);
    check_array(arr7_1, {-7, -3, 0, 1}, sizeof(arr7_1)/sizeof(int));
    int arr8_1[]={3, 3, 4, 0, 0, 0}, arr8_2[]={-5, -4, 5};
    merge_arrays(arr8_1, 3, arr8_2, 3);
    check_array(arr8_1, {-5, -4, 3, 3, 4, 5}, sizeof(arr8_1)/sizeof(int));
    int arr9_1[]={0, 0, 0}, arr9_2[]={-8, 0};
    merge_arrays(arr9_1, 1, arr9_2, 2);
    check_array(arr9_1, {-8, 0, 0}, sizeof(arr9_1)/sizeof(int));
    int arr10_1[]={-4, -4, 4, 0, 0}, arr10_2[]={-5, -4};
    merge_arrays(arr10_1, 3, arr10_2, 2);
    check_array(arr10_1, {-5, -4, -4, -4, 4}, sizeof(arr10_1)/sizeof(int));


    // 1.4
    std::cout << "1-4" << std::endl;
    int* res = pascal_triangle(1);
    check_array(res, {1}, 1);
    res = pascal_triangle( 2 );
    check_array(res, {1, 1}, 2);
    res = pascal_triangle( 3 );
    check_array(res, {1, 2, 1}, 3);
    res = pascal_triangle( 4 );
    check_array(res, {1, 3, 3, 1}, 4);
    res = pascal_triangle( 5 );
    check_array(res, {1, 4, 6, 4, 1}, 5);
    res = pascal_triangle( 6 );
    check_array(res, {1, 5, 10, 10, 5, 1}, 6);
    res = pascal_triangle( 7 );
    check_array(res, {1, 6, 15, 20, 15, 6, 1}, 7);
    res = pascal_triangle( 8 );
    check_array(res, {1, 7, 21, 35, 35, 21, 7, 1}, 8);
    res = pascal_triangle( 9 );
    check_array(res, {1, 8, 28, 56, 70, 56, 28, 8, 1}, 9);
    res = pascal_triangle( 10 );
    check_array(res, {1, 9, 36, 84, 126, 126, 84, 36, 9, 1}, 10);
    res = pascal_triangle( 11 );
    check_array(res, {1, 10, 45, 120, 210, 252, 210, 120, 45, 10, 1}, 11);
    res = pascal_triangle( 12 );
    check_array(res, {1, 11, 55, 165, 330, 462, 462, 330, 165, 55, 11, 1}, 12);
    res = pascal_triangle( 13 );
    check_array(res, {1, 12, 66, 220, 495, 792, 924, 792, 495, 220, 66, 12, 1}, 13);
    res = pascal_triangle( 14 );
    check_array(res, {1, 13, 78, 286, 715, 1287, 1716, 1716, 1287, 715, 286, 78, 13, 1}, 14);
    res = pascal_triangle( 15 );
    check_array(res, {1, 14, 91, 364, 1001, 2002, 3003, 3432, 3003, 2002, 1001, 364, 91, 14, 1}, 15);
    res = pascal_triangle( 16 );
    check_array(res, {1, 15, 105, 455, 1365, 3003, 5005, 6435, 6435, 5005, 3003, 1365, 455, 105, 15, 1}, 16);
    res = pascal_triangle( 17 );
    check_array(res, {1, 16, 120, 560, 1820, 4368, 8008, 11440, 12870, 11440, 8008, 4368, 1820, 560, 120, 16, 1}, 17);
    res = pascal_triangle( 18 );
    check_array(res, {1, 17, 136, 680, 2380, 6188, 12376, 19448, 24310, 24310, 19448, 12376, 6188, 2380, 680, 136, 17, 1}, 18);
    res = pascal_triangle( 19 );
    check_array(res, {1, 18, 153, 816, 3060, 8568, 18564, 31824, 43758, 48620, 43758, 31824, 18564, 8568, 3060, 816, 153, 18, 1}, 19);
    res = pascal_triangle( 20 );
    check_array(res, {1, 19, 171, 969, 3876, 11628, 27132, 50388, 75582, 92378, 92378, 75582, 50388, 27132, 11628, 3876, 969, 171, 19, 1}, 20);
    delete[] res;

    // 1.5
    std::cout << "1-5" << std::endl;
    int bills[] = {5,5,5,10,20};
    check_value(bibimbap_change(bills, sizeof(bills)/sizeof(int)), true);
    int bills2[] = {10, 5};
    check_value(bibimbap_change(bills2, sizeof(bills2)/sizeof(int)), false);
    int bills3[]={5, 20, 5, 5, 5};
    check_value(bibimbap_change(bills3, sizeof(bills3)/sizeof(int)), false);
    int bills4[]={5, 5, 5, 5, 20, 5, 10, 5};
    check_value(bibimbap_change(bills4, sizeof(bills4)/sizeof(int)), true);
    int bills5[]={20, 5, 20, 5, 10, 20, 20, 20, 5, 20, 5, 5};
    check_value(bibimbap_change(bills5, sizeof(bills5)/sizeof(int)), false);
    int bills6[]={10, 5, 20, 5, 10};
    check_value(bibimbap_change(bills6, sizeof(bills6)/sizeof(int)), false);
    int bills7[]={20, 5, 10, 5, 20, 5, 5, 5, 10, 10, 5, 5, 20, 5, 5, 5, 5, 20, 10};
    check_value(bibimbap_change(bills7, sizeof(bills7)/sizeof(int)), false);
    int bills8[]={5, 5, 20, 5, 10, 5};
    check_value(bibimbap_change(bills8, sizeof(bills8)/sizeof(int)), false);
    int bills9[]={5, 10, 5, 5, 5, 20, 5, 20};
    check_value(bibimbap_change(bills9, sizeof(bills9)/sizeof(int)), true);

    return 0;
}
