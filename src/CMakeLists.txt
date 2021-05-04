cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

find_package(Torch)

add_executable(SingleplayerDemo
	main.cpp
	"PlayerEntity.hpp"
	"PlayerEntity.cpp"
	"InputService.hpp"
	"InputService.cpp"
	"CastleEntity.cpp"
	"GameML.hpp"
        HealthBarComponent.cpp
	HealthBarComponent.hpp
	FireballEntity.cpp
	FireballEntity.hpp)

set_property(TARGET SingleplayerDemo PROPERTY CXX_STANDARD 17)

target_link_libraries(SingleplayerDemo Strife.Engine Strife.ML)

# Copy torch dlls
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_DIR}/lib/*.dll")

  add_custom_command(TARGET SingleplayerDemo
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
						${TORCH_DLLS}
						$<TARGET_FILE_DIR:SingleplayerDemo>)
endif()

add_custom_command(TARGET SingleplayerDemo
		POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_directory ../../assets
		$<TARGET_FILE_DIR:SingleplayerDemo>/assets)