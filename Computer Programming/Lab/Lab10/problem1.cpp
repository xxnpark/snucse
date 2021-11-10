#include <iostream>

using namespace std;

int main() {
    string name;
    cin >> name;
    if (name == "Youngki") {
        cout << "Hello, Professor!" << endl;
    } else {
        cout << "Hello, " + name + "!" << endl;
    }
    return 0;
}
