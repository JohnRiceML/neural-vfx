# native/core: device, queue, fence, command recording, resource helpers. OWNER: native solver engineer.
target_sources(nvfx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/context.cpp ${CMAKE_CURRENT_SOURCE_DIR}/texture2d.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/nvfx_core.h)
target_include_directories(nvfx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
