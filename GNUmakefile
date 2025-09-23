.PHONY: clear Debug Release

CMAKE := /opt/cmake-4.1.1/bin/cmake

format: ./src
	find src \
	-path src/ImGui -prune -o \
  -type f \( -iname '*.h' -o -iname '*.hpp' -o -iname '*.cpp' -o -iname '*.cu' -o -iname '*.glsl' \) \
  -exec clang-format -i {} +

clear:
	trash build && mkdir build && cd build

configure: clear
	${CMAKE} --preset aliu-configure

Debug:
	${CMAKE} --build --preset aliu-$@ && WAYLAND_DISPLAY='' XDG_SESSION_TYPE=x11 ./build/bin/$@/cis565_path_tracer ./scenes/sphere.json

Release:
	${CMAKE} --build --preset aliu-$@ && WAYLAND_DISPLAY='' XDG_SESSION_TYPE=x11 ./build/bin/$@/cis565_path_tracer ./scenes/sphere.json