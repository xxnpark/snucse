#include <iostream>

#define PI 3.14159
#define AREA(r) (PI * (r) * (r))

using namespace std;

int main() {
    float r;
    cin >> r;
    cout << AREA(r) << endl;
    return 0;
}
