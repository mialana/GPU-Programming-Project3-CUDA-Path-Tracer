
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/usd" "${CMAKE_SOURCE_DIR}/cmake/usd/packages" ${CMAKE_MODULE_PATH})

# appease some policies
cmake_policy(SET CMP0144 NEW)
###

# USD Config
set(BOOST_ROOT /usr)
set(TBB_ROOT /usr)

include(Options)
include(Defaults)
include(Packages)
include(Utils)

_usd_target_properties(${CMAKE_PROJECT_NAME} LIBRARIES ${GL_LIBRARIES} stream_compaction usd usdGeom gf vt hd)

target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE ENABLE_USD)