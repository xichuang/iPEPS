cmake_minimum_required(VERSION 2.8)
project(iPEPS)

set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_COMPILER icpc)

set(CMAKE_VERBOSE_MAKEFILE TRUE)
set(CMAKE_CXX_FLAGS "-m64 -std=c++11 -fPIC ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "-g3 -DNDEBUG -Wall -DPLATFORM_mkl -D__ASSERT_MACROS_DEFINE_VERSIONS_WITHOUT_UNDERSCORES=0")
#set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG -DPLATFORM_mkl -D__ASSERT_MACROS_DEFINE_VERSIONS_WITHOUT_UNDERSCORES=0")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")


include_directories(./ /home/xich/local/itensor-code
        /home/xich/local/include/eigen3
        /opt/intel/compilers_and_libraries/linux/mkl/include
        /home/xich/intel/mkl/include
        )
link_directories(/home/xich/local/itensor-code/lib
        /opt/intel/compilers_and_libraries/linux/mkl/lib/intel64
        /home/xich/intel/mkl/lib/intel64
        )

add_subdirectory(iPEPS)
add_subdirectory(utils)
add_subdirectory(tests)
add_subdirectory(BenchMarks)

