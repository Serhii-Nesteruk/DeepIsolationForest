cmake_minimum_required(VERSION 3.31)
project(DeepIForest)

set(CMAKE_CXX_STANDARD 20)

file(GLOB DIFIncludes CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h")

file(GLOB DIFSources CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")


add_executable(DeepIForest
    ${DIFSources}
)

target_include_directories(DeepIForest PRIVATE
    ${DIFIncludes}
)