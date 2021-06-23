cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

option(VCPKG_MANIFEST_MODE "vcpkg manifeset mode" ON)

set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake
		CACHE STRING "Vcpkg toolchain file")

project(SingleplayerDemo VERSION 1 LANGUAGES C CXX)

set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}")

if(MSVC)
	add_compile_options("/EHsc")
	add_compile_options("/MP")
	add_compile_options("/permissive")

	set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0 -DNOGDI -DWIN32")
endif()

add_subdirectory(Strife.ML)
add_subdirectory(Strife.Engine)
add_subdirectory(Strife.Common)
add_subdirectory(src)
