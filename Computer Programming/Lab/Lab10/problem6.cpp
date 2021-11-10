#include <iostream>

#define MAX_SIZE 100

using namespace std;

int main() {
    char s1[MAX_SIZE], s2[MAX_SIZE/2];
    char* s1_ptr = s1;
    char* s2_ptr = s2;

    cout << "write 1st word: " << endl;
    cin >> s1;
    cout << "write 2nd word: " << endl;
    cin >> s2;

    while (*(++s1_ptr));
    while (*(s1_ptr++) == *(s2_ptr++));

    cout << s1;

    return 0;
}