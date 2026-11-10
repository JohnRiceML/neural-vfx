# Top-level build of the native runtime. OWNER: lead.
# Each owner adds sources to the `nvfx` target (or its own executables) from its directory's CMakeLists.txt.
cmake_minimum_required(VERSION 3.24)
project(nvfx LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
foreach(kind RUNTIME LIBRARY ARCHIVE)
  set(CMAKE_${kind}_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endforeach()

# HLSL compiler from the Windows SDK (vcvars puts its bin directory in WindowsSdkVerBinPath).
find_program(NVFX_DXC dxc
  HINTS "$ENV{WindowsSdkVerBinPath}/x64" "$ENV{WindowsSdkDir}/bin/$ENV{WindowsSDKVersion}/x64"
  REQUIRED)
message(STATUS "dxc: ${NVFX_DXC}")

add_library(nvfx SHARED)
target_include_directories(nvfx PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(nvfx PRIVATE NVFX_BUILD NOMINMAX WIN32_LEAN_AND_MEAN)
target_compile_options(nvfx PRIVATE /W4 /permissive- /EHsc)
target_link_libraries(nvfx PRIVATE d3d12 dxgi dxguid)

foreach(dir core sim render infer app tests)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${dir}/CMakeLists.txt)
    add_subdirectory(${dir})
  endif()
endforeach()
