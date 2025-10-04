.PHONY: clear build Debug Release

CMAKE := /opt/cmake-4.1.1/bin/cmake

format: ./source
	find source \
  -type f \( -iname '*.h' -o -iname '*.hpp' -o -iname '*.cpp' -o -iname '*.cu' -o -iname '*.glsl' \) \
  -exec clang-format -i {} +

clear:
	trash build && mkdir build && cd build

configure: clear
	${CMAKE} --preset aliu-configure

build:
	${CMAKE} --build --preset aliu-$(MODE)

Debug:
	${CMAKE} --build --preset aliu-$@ && ./build/bin/$@/cis565_path_tracer ./scenes/sphere.json

Release:
	${CMAKE} --build --preset aliu-$@ && ./build/bin/$@/cis565_path_tracer ./scenes/sphere.json