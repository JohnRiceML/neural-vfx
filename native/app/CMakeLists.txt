# native/app: nvfx_demo.exe, the Gate 3 stand-alone windowed demo. OWNER: native app engineer.
# Windows SDK only (Win32, D3D12, DXGI); links the nvfx target. The DirectML / ONNX Runtime DLLs are already copied
# into <build>/bin by native/infer at configure time, next to nvfx.dll and this exe.
set(APP_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/gen)
file(MAKE_DIRECTORY ${APP_GEN_DIR})
set(APP_HLSL ${CMAKE_CURRENT_SOURCE_DIR}/present.hlsl)
set(APP_DXC_FLAGS -nologo -HV 2021 -O3 -WX)
add_custom_command(
  OUTPUT ${APP_GEN_DIR}/present_vs.h
  COMMAND ${NVFX_DXC} ${APP_DXC_FLAGS} -T vs_6_0 -E vs_main -Vn g_present_vs -Fh ${APP_GEN_DIR}/present_vs.h ${APP_HLSL}
  DEPENDS ${APP_HLSL}
  COMMENT "dxc present.hlsl:vs_main"
  VERBATIM)
add_custom_command(
  OUTPUT ${APP_GEN_DIR}/present_ps.h
  COMMAND ${NVFX_DXC} ${APP_DXC_FLAGS} -T ps_6_0 -E ps_main -Vn g_present_ps -Fh ${APP_GEN_DIR}/present_ps.h ${APP_HLSL}
  DEPENDS ${APP_HLSL}
  COMMENT "dxc present.hlsl:ps_main"
  VERBATIM)

add_executable(nvfx_demo
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hud_font.h
  ${APP_GEN_DIR}/present_vs.h
  ${APP_GEN_DIR}/present_ps.h)
# public headers currently live in three places: include/ (via the nvfx target), render/ and infer/
target_include_directories(nvfx_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${APP_GEN_DIR}
                           ${CMAKE_CURRENT_SOURCE_DIR}/../render ${CMAKE_CURRENT_SOURCE_DIR}/../infer)
target_link_libraries(nvfx_demo PRIVATE nvfx d3d12 dxgi dxguid user32)
target_compile_options(nvfx_demo PRIVATE /W4 /permissive- /EHsc)
target_compile_definitions(nvfx_demo PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN UNICODE _UNICODE _CRT_SECURE_NO_WARNINGS
                           _WIN32_WINNT=0x0A00)
