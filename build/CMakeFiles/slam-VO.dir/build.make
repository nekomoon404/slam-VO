# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/pty/slambook/homework/yxc

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/pty/slambook/homework/yxc/build

# Include any dependencies generated for this target.
include CMakeFiles/slam-VO.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/slam-VO.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/slam-VO.dir/flags.make

CMakeFiles/slam-VO.dir/src/slam-VO.cpp.o: CMakeFiles/slam-VO.dir/flags.make
CMakeFiles/slam-VO.dir/src/slam-VO.cpp.o: ../src/slam-VO.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pty/slambook/homework/yxc/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/slam-VO.dir/src/slam-VO.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/slam-VO.dir/src/slam-VO.cpp.o -c /home/pty/slambook/homework/yxc/src/slam-VO.cpp

CMakeFiles/slam-VO.dir/src/slam-VO.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/slam-VO.dir/src/slam-VO.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pty/slambook/homework/yxc/src/slam-VO.cpp > CMakeFiles/slam-VO.dir/src/slam-VO.cpp.i

CMakeFiles/slam-VO.dir/src/slam-VO.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/slam-VO.dir/src/slam-VO.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pty/slambook/homework/yxc/src/slam-VO.cpp -o CMakeFiles/slam-VO.dir/src/slam-VO.cpp.s

CMakeFiles/slam-VO.dir/src/slam-VO.cpp.o.requires:

.PHONY : CMakeFiles/slam-VO.dir/src/slam-VO.cpp.o.requires

CMakeFiles/slam-VO.dir/src/slam-VO.cpp.o.provides: CMakeFiles/slam-VO.dir/src/slam-VO.cpp.o.requires
	$(MAKE) -f CMakeFiles/slam-VO.dir/build.make CMakeFiles/slam-VO.dir/src/slam-VO.cpp.o.provides.build
.PHONY : CMakeFiles/slam-VO.dir/src/slam-VO.cpp.o.provides

CMakeFiles/slam-VO.dir/src/slam-VO.cpp.o.provides.build: CMakeFiles/slam-VO.dir/src/slam-VO.cpp.o


# Object files for target slam-VO
slam__VO_OBJECTS = \
"CMakeFiles/slam-VO.dir/src/slam-VO.cpp.o"

# External object files for target slam-VO
slam__VO_EXTERNAL_OBJECTS =

slam-VO: CMakeFiles/slam-VO.dir/src/slam-VO.cpp.o
slam-VO: CMakeFiles/slam-VO.dir/build.make
slam-VO: /usr/local/lib/libopencv_dnn.so.3.4.0
slam-VO: /usr/local/lib/libopencv_ml.so.3.4.0
slam-VO: /usr/local/lib/libopencv_objdetect.so.3.4.0
slam-VO: /usr/local/lib/libopencv_shape.so.3.4.0
slam-VO: /usr/local/lib/libopencv_stitching.so.3.4.0
slam-VO: /usr/local/lib/libopencv_superres.so.3.4.0
slam-VO: /usr/local/lib/libopencv_videostab.so.3.4.0
slam-VO: /usr/local/lib/libopencv_viz.so.3.4.0
slam-VO: /usr/lib/x86_64-linux-gnu/libcxsparse.so
slam-VO: /usr/local/lib/libopencv_calib3d.so.3.4.0
slam-VO: /usr/local/lib/libopencv_features2d.so.3.4.0
slam-VO: /usr/local/lib/libopencv_flann.so.3.4.0
slam-VO: /usr/local/lib/libopencv_highgui.so.3.4.0
slam-VO: /usr/local/lib/libopencv_photo.so.3.4.0
slam-VO: /usr/local/lib/libopencv_video.so.3.4.0
slam-VO: /usr/local/lib/libopencv_videoio.so.3.4.0
slam-VO: /usr/local/lib/libopencv_imgcodecs.so.3.4.0
slam-VO: /usr/local/lib/libopencv_imgproc.so.3.4.0
slam-VO: /usr/local/lib/libopencv_core.so.3.4.0
slam-VO: CMakeFiles/slam-VO.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/pty/slambook/homework/yxc/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable slam-VO"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/slam-VO.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/slam-VO.dir/build: slam-VO

.PHONY : CMakeFiles/slam-VO.dir/build

CMakeFiles/slam-VO.dir/requires: CMakeFiles/slam-VO.dir/src/slam-VO.cpp.o.requires

.PHONY : CMakeFiles/slam-VO.dir/requires

CMakeFiles/slam-VO.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/slam-VO.dir/cmake_clean.cmake
.PHONY : CMakeFiles/slam-VO.dir/clean

CMakeFiles/slam-VO.dir/depend:
	cd /home/pty/slambook/homework/yxc/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/pty/slambook/homework/yxc /home/pty/slambook/homework/yxc /home/pty/slambook/homework/yxc/build /home/pty/slambook/homework/yxc/build /home/pty/slambook/homework/yxc/build/CMakeFiles/slam-VO.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/slam-VO.dir/depend

