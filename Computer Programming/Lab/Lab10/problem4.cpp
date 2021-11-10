#include <iostream>
#include <fstream>

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
    int n;
    string str;

    ifstream input_file("input.txt");
    ofstream output_file("output.txt");

    while (getline(input_file, str)) {
        n = stoi(str);
        output_file << n << " : " << is_prime(n) << endl;
    }

    input_file.close();
    output_file.close();
    return 0;
}
