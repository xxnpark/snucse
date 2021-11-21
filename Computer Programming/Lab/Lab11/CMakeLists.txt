cmake_minimum_required(VERSION 3.20)
project(Lab11)

set(CMAKE_CXX_STANDARD 14)

add_executable(Lab11 main.cpp Point.h Grid.h Point.cpp Grid.cpp)

add_executable(main main.cpp)