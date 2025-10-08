.PHONY: clear build Debug Release

CMAKE := /opt/cmake-4.1.1/bin/cmake

format: ./src
	find src stream_compaction \
	-path src/ImGui -prune -o \
  -type f \( -iname '*.h' -o -iname '*.hpp' -o -iname '*.cpp' -o -iname '*.cu' -o -iname '*.glsl' \) \
  -exec clang-format -i {} +

clear:
	trash build && mkdir build && cd build

configure: clear
	${CMAKE} --preset aliu-configure

build:
	${CMAKE} --build --preset aliu-$(MODE)

Debug:
	${CMAKE} --build --preset aliu-$@ && ./build/bin/$@/cuda_path_tracer ./scenes/jelloShelf.usda

Release:
	${CMAKE} --build --preset aliu-$@ && ./build/bin/$@/cuda_path_tracer ./scenes/jelloShelf.usda