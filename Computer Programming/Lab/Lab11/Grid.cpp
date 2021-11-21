#include "Grid.h"

Grid::Grid(int row, int column): row(row), column(column) {
    grid = new int*[row];
    for (int i = 0; i < row; i++) {
        grid[i] = new int[column];
    }
    initialize_with_zeros();
}

void Grid::initialize_with_zeros() {
    for (int r = 0; r < row; r++) {
        for (int c = 0; c < column; c++) {
            grid[r][c] = 0;
        }
    }
}

void Grid::mark_point(Point p1) {
    int x = p1.getX();
    int y = p1.getY();

    if (x >= row || y >= column) {
        return;
    }

    grid[x][y] = mark_counter++;
}

int Grid::getRow() const {
    return row;
}

int Grid::getColumn() const {
    return column;
}

int Grid::getAt(int r, int c) const {
    return grid[r][c];
}

void Grid::setAt(int r, int c, int v) {
    grid[r][c] = v;
}

void Grid::printGrid() {
    std::cout << "grid : \n";
    for (int r = 0; r < row; r++) {
        for (int c = 0; c < column; c++) {
            std::cout << grid[r][c] << " ";
        }
        std::cout << std::endl;
    }
}

Grid::Grid(int row, int column, int** grid_value): row(row), column(column) {
    grid = new int*[row];
    for (int i = 0; i < row; i++) {
        grid[i] = new int[column];
        for (int j = 0; j < column; j++) {
            grid[i][j] = grid_value[i][j];
        }
    }
}

Grid::Grid(Grid const &g): Grid(g.row, g.column, g.grid) {}

Grid::~Grid() {
    std::cout << "Clean-up Grid" << std::endl;
    for (int i = 0; i < row; i++) {
        delete[] grid[i];
    }
    delete[] grid;
}