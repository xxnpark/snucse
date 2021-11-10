#include <iostream>

using namespace std;

bool is_prime(int n) {
    for (int i = 2; i * i <= n; i++) {
        if (n % i == 0) {
            return false;
        }
    }
    return true;
}

int main() {
    for (int i = 2; i < 100; i++) {
        cout << i << " : " << is_prime(i) << endl;
    }
    return 0;
}
