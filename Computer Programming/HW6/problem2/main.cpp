#include <iostream>
#include <fstream>
#include "config.h"
#include "app.h"

int main() {
    App app(std::cin, std::cout);
    app.run();
}
