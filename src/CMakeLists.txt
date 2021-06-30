cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

find_package(Torch)

add_executable(SingleplayerDemo
	"main.cpp"
	"PlayerEntity.hpp"
	"PlayerEntity.cpp"
	"InputService.hpp"
	"InputService.cpp"
	"CastleEntity.cpp"
	"GameML.hpp"
    "HealthBarComponent.cpp"
	"HealthBarComponent.hpp"
	"FireballEntity.cpp"
	"FireballEntity.hpp"
	"PlayerNeuralNetworkService.hpp"
	"PlayerNeuralNetworkService.cpp")

set_property(TARGET SingleplayerDemo PROPERTY CXX_STANDARD 17)

target_link_libraries(SingleplayerDemo Strife.Engine Strife.ML)

add_custom_command(TARGET SingleplayerDemo
		POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/assets $<TARGET_FILE_DIR:SingleplayerDemo>/assets)