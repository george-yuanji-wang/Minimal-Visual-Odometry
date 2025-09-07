cmake_minimum_required(VERSION 3.15)
project(vo_core)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --------- Dependencies ----------
find_package(OpenCV REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

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
    /opt/homebrew/include   
)

target_link_libraries(${PROJECT_NAME}
  PUBLIC
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

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
