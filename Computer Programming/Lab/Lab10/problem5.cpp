#include <iostream>

using namespace std;

void three_swap(int* a, int* b, int *c) {
    int temp_a = *a;
    *a = *b;
    *b = *c;
    *c = temp_a;
}

void three_swap(int& a, int& b, int& c) {
    int temp_a = a;
    a = b;
    b = c;
    c = temp_a;
}

int main() {
    int a, b, c;
    cin >> a;
    cin >> b;
    cin >> c;

    cout << "original values: " << a << b << c << endl;

    three_swap(&a, &b, &c);
    cout << "swap once: " << a << b << c << endl;

    three_swap(a, b, c) ;
    cout << "swap twice: " << a << b << c << endl;

    return 0;
}
