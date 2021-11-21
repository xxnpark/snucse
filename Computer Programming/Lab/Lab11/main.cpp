#include <iostream>
#include "Point.h"
#include "Grid.h"

void printNumberGrid(Grid g){
    for (int r = 0; r < g.getRow(); r++) {
        for (int c = 0; c < g.getColumn(); c++) {
            g.setAt(r, c, r * g.getColumn() + c);
        }
    }
    g.printGrid();
}

int main() {
    Grid g(2,3);

    g.printGrid();

    Point p1(1, 0);
    Point p2(0, 1);
    Point p3(3, 3);

    g.mark_point(p1);
    g.mark_point(p2);
    g.mark_point(p3);

    g.printGrid();

    return 0;
}
