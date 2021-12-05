#include <string>

bool is_palindrome(std::string s) {
    if (s.length() <= 1) {
        return true;
    }

    for (int i = 0; i <= s.length() / 2; i++) {
        if (s[i] != s[s.length() - i - 1]) {
            return false;
        }
    }

    return true;
}