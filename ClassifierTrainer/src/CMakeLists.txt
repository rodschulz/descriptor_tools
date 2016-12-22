cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

# Project's name
project(DescriptorApps)

# Enable or disable the use of OpenMP
set(useOpenMP FALSE)

# Colors defined for messages
if(NOT WIN32)
	string(ASCII 27 Esc)
	set(ColorReset	"${Esc}[m")
	set(Red 		"${Esc}[31m")
	set(Green 		"${Esc}[32m")
	set(Yellow 		"${Esc}[33m")
	set(Blue 		"${Esc}[34m")
	set(Magenta 	"${Esc}[35m")
	set(Cyan 		"${Esc}[36m")
endif()

# Add additional path for cmake to find OpenCV
set(CMAKE_PREFIX_PATH "/opt/ros/hydro/" ${CMAKE_PREFIX_PATH})

# Find required packages
find_package(PCL 1.7 REQUIRED)
find_package(OpenCV 2.4.8 REQUIRED)
find_package(yaml-cpp 0.5.2 REQUIRED)
find_package(Boost 1.46 COMPONENTS unit_test_framework REQUIRED)

# Print information related to libraries
message(STATUS "${Yellow}PCL found version ${PCL_VERSION}${ColorReset}")
message(STATUS "${Cyan}\tPCL_INCLUDE_DIRS = ${PCL_INCLUDE_DIRS}${ColorReset}")

message(STATUS "${Yellow}Found OpenCV version ${OpenCV_VERSION}${ColorReset}")
message(STATUS "${Cyan}\tOpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}${ColorReset}")

message(STATUS "${Yellow}Found yaml-cpp version ${yaml-cpp_VERSION}${ColorReset}")
message(STATUS "${Cyan}\tYAML_CPP_INCLUDE_DIR = ${YAML_CPP_INCLUDE_DIR}${ColorReset}")

message(STATUS "${Yellow}Found Boost version ${Boost_LIB_VERSION}${ColorReset}")
message(STATUS "${Cyan}\tBoost_INCLUDE_DIRS = ${Boost_INCLUDE_DIRS}${ColorReset}")

# Add relevant linking directories
link_directories(${PCL_LIBRARY_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

# Set the build type for PCL
set(PCL_BUILD_TYPE Release)

# Add extra required definitions
add_definitions(${PCL_DEFINITIONS})
add_definitions("-Wall")

include_directories(./include/)

# Set the linking dependencies for all the applications
set(APPS_LINKING_DEPENDENCIES
	${PCL_LIBRARIES}
	${YAML_CPP_LIBRARIES}
	${OpenCV_LIBS})

file(GLOB TRAINER_SRC "Trainer.cpp")
add_executable(Trainer ${TRAINER_SRC})
target_link_libraries(Trainer ${APPS_LINKING_DEPENDENCIES})