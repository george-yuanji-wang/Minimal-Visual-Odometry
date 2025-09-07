cmake_minimum_required(VERSION 3.15)
project(vo_core)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --------- Dependencies ----------
find_package(OpenCV REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Where Homebrew typically puts g2o:
set(G2O_HINT_PREFIXES
  /opt/homebrew/opt/g2o
  /opt/homebrew
  /usr/local/opt/g2o
  /usr/local
)

# Headers
find_path(G2O_INCLUDE_DIR
  NAMES g2o/core/sparse_optimizer.h
  HINTS ${G2O_HINT_PREFIXES}
  PATH_SUFFIXES include
  REQUIRED
)

# Core/stuff + solvers
find_library(G2O_CORE         NAMES g2o_core         HINTS ${G2O_HINT_PREFIXES} PATH_SUFFIXES lib REQUIRED)
find_library(G2O_STUFF        NAMES g2o_stuff        HINTS ${G2O_HINT_PREFIXES} PATH_SUFFIXES lib REQUIRED)
find_library(G2O_SOLVER_EIGEN NAMES g2o_solver_eigen HINTS ${G2O_HINT_PREFIXES} PATH_SUFFIXES lib REQUIRED)
find_library(G2O_SOLVER_DENSE NAMES g2o_solver_dense HINTS ${G2O_HINT_PREFIXES} PATH_SUFFIXES lib REQUIRED)

# Type modules you actually use
find_library(G2O_TYPES_SBA    NAMES g2o_types_sba    HINTS ${G2O_HINT_PREFIXES} PATH_SUFFIXES lib)
find_library(G2O_TYPES_SLAM3D NAMES g2o_types_slam3d HINTS ${G2O_HINT_PREFIXES} PATH_SUFFIXES lib)

# You include both SBA (camera proj edges) AND SLAM3D (VertexPointXYZ), so at least one must exist,
# and ideally both.
if (NOT G2O_TYPES_SBA AND NOT G2O_TYPES_SLAM3D)
  message(FATAL_ERROR "Neither g2o_types_sba nor g2o_types_slam3d found. Install g2o with SBA and/or SLAM3D types.")
endif()

message(STATUS "g2o include: ${G2O_INCLUDE_DIR}")
message(STATUS "g2o libs:
  CORE=${G2O_CORE}
  STUFF=${G2O_STUFF}
  EIGEN_SOLVER=${G2O_SOLVER_EIGEN}
  DENSE_SOLVER=${G2O_SOLVER_DENSE}
  TYPES_SBA=${G2O_TYPES_SBA}
  TYPES_SLAM3D=${G2O_TYPES_SLAM3D}
")

# --------- Library ----------
add_library(${PROJECT_NAME}
  src/vo_core/feature_extractor.cpp
  src/vo_core/feature_matcher.cpp
  src/vo_core/pose_estimator.cpp
  src/vo_core/vo_pipeline.cpp
  src/geometry/geometry.cpp
)

target_include_directories(${PROJECT_NAME}
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${G2O_INCLUDE_DIR}
    /opt/homebrew/include            
)

target_link_libraries(${PROJECT_NAME}
  PUBLIC
    ${OpenCV_LIBS}
    Eigen3::Eigen
    ${G2O_CORE}
    ${G2O_STUFF}
    ${G2O_SOLVER_EIGEN}
    ${G2O_SOLVER_DENSE}
)

if (G2O_TYPES_SBA)
  target_link_libraries(${PROJECT_NAME} PUBLIC ${G2O_TYPES_SBA})
endif()
if (G2O_TYPES_SLAM3D)
  target_link_libraries(${PROJECT_NAME} PUBLIC ${G2O_TYPES_SLAM3D})
endif()

# --------- Executables ----------
add_executable(test_feature test/test_feature.cpp)
target_link_libraries(test_feature PRIVATE vo_core)

add_executable(cam_feature test/cam_feature.cpp)
target_link_libraries(cam_feature PRIVATE vo_core)

add_executable(test_matcher test/test_matcher.cpp)
target_link_libraries(test_matcher PRIVATE vo_core)

add_executable(calibrate src/calibrate.cpp)
target_link_libraries(calibrate PRIVATE vo_core)

add_executable(tum_vo_only test/tum_vo_only.cpp)
target_link_libraries(tum_vo_only PRIVATE vo_core)

add_executable(parameters_tune test/parameters_tune.cpp)
target_link_libraries(parameters_tune PRIVATE vo_core)

