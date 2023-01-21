# message(STATUS <test>)
cmake_minimum_required(VERSION 3.24)
project(DroneStitcher)

set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_FIND_DEBUG_MODE 1) # Enable for debugging

# Vulkan is manually installed on my System. The environment variable VULKAN_SDK also had to be defined manually:
#     launchctl setenv VULKAN_SDK /opt/VulkanSDK/1.3.236.0
# alternatively in here:
#     set(VULKAN_SDK /opt/VulkanSDK/1.3.236.0)

set(PROJECT_INCLUDE_DIR include)
set(VULKAN_INCLUDE_DIR /opt/VulkanSDK/1.3.236.0/include)
set(IMGUI_INCLUDE_DIR include/imgui)

set(PROJECT_SOURCE_DIR src)
set(IMGUI_SOURCE_DIR include/imgui)
set(VULKAN_SOURCE_DIR ${VULKAN_SDK}/include)

file(GLOB SOURCE_FILES ${PROJECT_SOURCE_DIR}/*.cpp ${IMGUI_SOURCE_DIR}/*.cpp)
message(STATUS ${SOURCE_FILES})
include_directories(${PROJECT_INCLUDE_DIR} ${VULKAN_INCLUDE_DIR} ${IMGUI_INCLUDE_DIR})

set(PROJECT_SOURCES ${SOURCE_FILES}
                    ${IMGUI_SOURCEFILES})

find_package(glm REQUIRED)
find_package(glfw3 REQUIRED)
find_package(Vulkan REQUIRED)
#find_package(imgui REQUIRED)

add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})
target_link_libraries(${PROJECT_NAME} Vulkan::Vulkan glm::glm glfw)
