
bool bibimbap_change(int* bills, int N) {
    int five = 0, ten = 0, twenty = 0;
    for (int i = 0; i < N; i++) {
        if (bills[i] == 5) {
            five++;
        } else if (bills[i] == 10) {
            if (five > 0) {
                five--;
                ten++;
            } else {
                return false;
            }
        } else {
            if (five > 2) {
                five--; five--; five--;
                twenty++;
            } else if (five > 0 && ten > 0) {
                five--;
                ten--;
                twenty++;
            } else {
                return false;
            }
        }
    }
    return true;
}

