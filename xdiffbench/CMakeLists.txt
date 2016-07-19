cmake_minimum_required(VERSION 3.5)
project(xdiffbench)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Werror -march=native")

set(SOURCE_FILES main.cpp)

find_package(PkgConfig)
pkg_check_modules(PC_LIBXDIFF QUIET libxdiff)
set(LIBXDIFF_DEFINITIONS ${PC_LIBXML_CFLAGS_OTHER})
find_path(LIBXDIFF_INCLUDE_DIR libxml/xpath.h
          HINTS ${PC_LIBXML_INCLUDEDIR} ${PC_LIBXML_INCLUDE_DIRS}
          PATH_SUFFIXES libxdiff )
find_library(LIBXDIFF_LIBRARY NAMES xdiff libxdiff
             HINTS ${PC_LIBXML_LIBDIR} ${PC_LIBXML_LIBRARY_DIRS} )
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LibXdiff  DEFAULT_MSG
                                  LIBXDIFF_LIBRARY LIBXDIFF_INCLUDE_DIR)
mark_as_advanced(LIBXDIFF_INCLUDE_DIR LIBXDIFF_LIBRARY )
set(LIBXDIFF_LIBRARIES ${LIBXDIFF_LIBRARY} )
set(LIBXDIFF_INCLUDE_DIRS ${LIBXDIFF_INCLUDE_DIR} )

add_executable(xdiffbench ${SOURCE_FILES})
target_link_libraries(xdiffbench xdiff)