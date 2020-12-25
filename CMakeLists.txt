cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

project(SingleplayerDemo VERSION 1 LANGUAGES C CXX)

set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}")

add_subdirectory(src)
add_subdirectory(Strife.Engine)
