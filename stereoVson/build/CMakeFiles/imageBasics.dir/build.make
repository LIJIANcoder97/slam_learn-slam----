# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

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
CMAKE_SOURCE_DIR = /home/q-v-/文档/slam_learn

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/q-v-/文档/slam_learn/build

# Include any dependencies generated for this target.
include CMakeFiles/imageBasics.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/imageBasics.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/imageBasics.dir/flags.make

CMakeFiles/imageBasics.dir/imageBasic.cpp.o: CMakeFiles/imageBasics.dir/flags.make
CMakeFiles/imageBasics.dir/imageBasic.cpp.o: ../imageBasic.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/q-v-/文档/slam_learn/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/imageBasics.dir/imageBasic.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/imageBasics.dir/imageBasic.cpp.o -c /home/q-v-/文档/slam_learn/imageBasic.cpp

CMakeFiles/imageBasics.dir/imageBasic.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/imageBasics.dir/imageBasic.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/q-v-/文档/slam_learn/imageBasic.cpp > CMakeFiles/imageBasics.dir/imageBasic.cpp.i

CMakeFiles/imageBasics.dir/imageBasic.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/imageBasics.dir/imageBasic.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/q-v-/文档/slam_learn/imageBasic.cpp -o CMakeFiles/imageBasics.dir/imageBasic.cpp.s

# Object files for target imageBasics
imageBasics_OBJECTS = \
"CMakeFiles/imageBasics.dir/imageBasic.cpp.o"

# External object files for target imageBasics
imageBasics_EXTERNAL_OBJECTS =

imageBasics: CMakeFiles/imageBasics.dir/imageBasic.cpp.o
imageBasics: CMakeFiles/imageBasics.dir/build.make
imageBasics: /usr/local/lib/libopencv_calib3d.so.4.7.0
imageBasics: /usr/local/lib/libopencv_core.so.4.7.0
imageBasics: /usr/local/lib/libopencv_dnn.so.4.7.0
imageBasics: /usr/local/lib/libopencv_features2d.so.4.7.0
imageBasics: /usr/local/lib/libopencv_flann.so.4.7.0
imageBasics: /usr/local/lib/libopencv_gapi.so.4.7.0
imageBasics: /usr/local/lib/libopencv_highgui.so.4.7.0
imageBasics: /usr/local/lib/libopencv_imgcodecs.so.4.7.0
imageBasics: /usr/local/lib/libopencv_imgproc.so.4.7.0
imageBasics: /usr/local/lib/libopencv_ml.so.4.7.0
imageBasics: /usr/local/lib/libopencv_objdetect.so.4.7.0
imageBasics: /usr/local/lib/libopencv_photo.so.4.7.0
imageBasics: /usr/local/lib/libopencv_stitching.so.4.7.0
imageBasics: /usr/local/lib/libopencv_video.so.4.7.0
imageBasics: /usr/local/lib/libopencv_videoio.so.4.7.0
imageBasics: /usr/local/lib/libopencv_imgcodecs.so.4.7.0
imageBasics: /usr/local/lib/libopencv_dnn.so.4.7.0
imageBasics: /usr/local/lib/libopencv_calib3d.so.4.7.0
imageBasics: /usr/local/lib/libopencv_features2d.so.4.7.0
imageBasics: /usr/local/lib/libopencv_flann.so.4.7.0
imageBasics: /usr/local/lib/libopencv_imgproc.so.4.7.0
imageBasics: /usr/local/lib/libopencv_core.so.4.7.0
imageBasics: CMakeFiles/imageBasics.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/q-v-/文档/slam_learn/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable imageBasics"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/imageBasics.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/imageBasics.dir/build: imageBasics

.PHONY : CMakeFiles/imageBasics.dir/build

CMakeFiles/imageBasics.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/imageBasics.dir/cmake_clean.cmake
.PHONY : CMakeFiles/imageBasics.dir/clean

CMakeFiles/imageBasics.dir/depend:
	cd /home/q-v-/文档/slam_learn/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/q-v-/文档/slam_learn /home/q-v-/文档/slam_learn /home/q-v-/文档/slam_learn/build /home/q-v-/文档/slam_learn/build /home/q-v-/文档/slam_learn/build/CMakeFiles/imageBasics.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/imageBasics.dir/depend

