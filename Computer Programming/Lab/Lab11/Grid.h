#include <iostream>
#include "Point.h"

#ifndef GRID_H
#define GRID_H

class Grid {
    int **grid;
    int row, column;
    int mark_counter = 1;

public:
    Grid(int row, int column);
    Grid(int row, int column, int** grid_value);
    Grid(Grid const &g);

    void initialize_with_zeros();

    void mark_point(Point p1);

    int getRow() const;
    int getColumn() const;
    int getAt(int r, int c) const;
    void setAt(int r, int c, int v);

    void printGrid();

    ~Grid();
};

#endif
