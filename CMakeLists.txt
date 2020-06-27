cmake_minimum_required(VERSION 2.8)
project( slam-VO )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -march=native -O3" )

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )


############### dependencies ######################
# Eigen
include_directories( "/usr/include/eigen3" )

# OpenCV
find_package( OpenCV  REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# G2O
find_package( G2O REQUIRED )
include_directories( ${G2O_INCLUDE_DIRS} )

#CSparse
find_package( CSparse REQUIRED)
include_directories( ${CSPARSE_INCLUDE_DIR} )


file(GLOB variables
    "src/*.h"
    "src/*.cpp"
)
add_executable( slam-VO ${variables} )
target_link_libraries( slam-VO 
   ${OpenCV_LIBS}
   ${CSPARSE_LIBRARY}
   g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension)
