cmake_minimum_required(VERSION 3.24)
project(DroneStitcher)

set(CMAKE_CXX_STANDARD 20)

set(PROJECT_INCLUDE_DIR include)
set(PROJECT_SOURCE_DIR src)
set(IMGUI_DIR include/imgui)

file(GLOB IMGUI_SOURCEFILES ${IMGUI_DIR}/*.cpp)
file(GLOB SOURCE_FILES ${PROJECT_SOURCE_DIR}/*.cpp ${PROJECT_SOURCE_DIR}/*.c)

include_directories(${PROJECT_INCLUDE_DIR} ${IMGUI_DIR})

set(PROJECT_SOURCES ${SOURCE_FILES}
                    ${IMGUI_SOURCEFILES})

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
#find_package(glad REQUIRED)
#find_package(glm REQUIRED)
#find_package(freetype REQUIRED)
#find_package(imgui REQUIRED)

add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})
target_link_libraries(${PROJECT_NAME} ${OPENGL_gl_LIBRARY} glfw)
