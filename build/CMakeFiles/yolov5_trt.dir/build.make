# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /media/yao/Data/yolov5-tensorrt

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /media/yao/Data/yolov5-tensorrt/build

# Include any dependencies generated for this target.
include CMakeFiles/yolov5_trt.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/yolov5_trt.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/yolov5_trt.dir/flags.make

CMakeFiles/yolov5_trt.dir/main.cpp.o: CMakeFiles/yolov5_trt.dir/flags.make
CMakeFiles/yolov5_trt.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/yao/Data/yolov5-tensorrt/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/yolov5_trt.dir/main.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/yolov5_trt.dir/main.cpp.o -c /media/yao/Data/yolov5-tensorrt/main.cpp

CMakeFiles/yolov5_trt.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/yolov5_trt.dir/main.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/yao/Data/yolov5-tensorrt/main.cpp > CMakeFiles/yolov5_trt.dir/main.cpp.i

CMakeFiles/yolov5_trt.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/yolov5_trt.dir/main.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/yao/Data/yolov5-tensorrt/main.cpp -o CMakeFiles/yolov5_trt.dir/main.cpp.s

CMakeFiles/yolov5_trt.dir/src/yolov5.cpp.o: CMakeFiles/yolov5_trt.dir/flags.make
CMakeFiles/yolov5_trt.dir/src/yolov5.cpp.o: ../src/yolov5.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/yao/Data/yolov5-tensorrt/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/yolov5_trt.dir/src/yolov5.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/yolov5_trt.dir/src/yolov5.cpp.o -c /media/yao/Data/yolov5-tensorrt/src/yolov5.cpp

CMakeFiles/yolov5_trt.dir/src/yolov5.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/yolov5_trt.dir/src/yolov5.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/yao/Data/yolov5-tensorrt/src/yolov5.cpp > CMakeFiles/yolov5_trt.dir/src/yolov5.cpp.i

CMakeFiles/yolov5_trt.dir/src/yolov5.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/yolov5_trt.dir/src/yolov5.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/yao/Data/yolov5-tensorrt/src/yolov5.cpp -o CMakeFiles/yolov5_trt.dir/src/yolov5.cpp.s

# Object files for target yolov5_trt
yolov5_trt_OBJECTS = \
"CMakeFiles/yolov5_trt.dir/main.cpp.o" \
"CMakeFiles/yolov5_trt.dir/src/yolov5.cpp.o"

# External object files for target yolov5_trt
yolov5_trt_EXTERNAL_OBJECTS =

yolov5_trt: CMakeFiles/yolov5_trt.dir/main.cpp.o
yolov5_trt: CMakeFiles/yolov5_trt.dir/src/yolov5.cpp.o
yolov5_trt: CMakeFiles/yolov5_trt.dir/build.make
yolov5_trt: /usr/local/lib/libopencv_gapi.so.4.5.1
yolov5_trt: /usr/local/lib/libopencv_highgui.so.4.5.1
yolov5_trt: /usr/local/lib/libopencv_ml.so.4.5.1
yolov5_trt: /usr/local/lib/libopencv_objdetect.so.4.5.1
yolov5_trt: /usr/local/lib/libopencv_photo.so.4.5.1
yolov5_trt: /usr/local/lib/libopencv_stitching.so.4.5.1
yolov5_trt: /usr/local/lib/libopencv_video.so.4.5.1
yolov5_trt: /usr/local/lib/libopencv_videoio.so.4.5.1
yolov5_trt: /usr/local/cuda/lib64/libcudart_static.a
yolov5_trt: /usr/lib/x86_64-linux-gnu/librt.so
yolov5_trt: /home/yao/TensorRT-7.1.3.4/lib/libnvinfer.so
yolov5_trt: /home/yao/TensorRT-7.1.3.4/lib/libnvonnxparser.so
yolov5_trt: /usr/local/lib/libopencv_dnn.so.4.5.1
yolov5_trt: /usr/local/lib/libopencv_imgcodecs.so.4.5.1
yolov5_trt: /usr/local/lib/libopencv_calib3d.so.4.5.1
yolov5_trt: /usr/local/lib/libopencv_features2d.so.4.5.1
yolov5_trt: /usr/local/lib/libopencv_flann.so.4.5.1
yolov5_trt: /usr/local/lib/libopencv_imgproc.so.4.5.1
yolov5_trt: /usr/local/lib/libopencv_core.so.4.5.1
yolov5_trt: CMakeFiles/yolov5_trt.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/media/yao/Data/yolov5-tensorrt/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable yolov5_trt"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/yolov5_trt.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/yolov5_trt.dir/build: yolov5_trt

.PHONY : CMakeFiles/yolov5_trt.dir/build

CMakeFiles/yolov5_trt.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/yolov5_trt.dir/cmake_clean.cmake
.PHONY : CMakeFiles/yolov5_trt.dir/clean

CMakeFiles/yolov5_trt.dir/depend:
	cd /media/yao/Data/yolov5-tensorrt/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /media/yao/Data/yolov5-tensorrt /media/yao/Data/yolov5-tensorrt /media/yao/Data/yolov5-tensorrt/build /media/yao/Data/yolov5-tensorrt/build /media/yao/Data/yolov5-tensorrt/build/CMakeFiles/yolov5_trt.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/yolov5_trt.dir/depend
