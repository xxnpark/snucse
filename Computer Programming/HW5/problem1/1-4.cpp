
int C(int n, int k) {
    int ret = 1;

    for (int i = 1; i <= k; i++) {
        ret = ret * (n-i+1) / i;
    }

    return ret;
}

int* pascal_triangle(int N) {
    int* pascal_Nth = new int();

    for (int i = 0; i < N; i++) {
        pascal_Nth[i] = C(N - 1, i);
    }

    return pascal_Nth;
}

