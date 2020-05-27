# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

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
CMAKE_SOURCE_DIR = /home/lc432959/repos/hpc/llvm-openmp/runtime

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lc432959/repos/hpc/llvm-openmp/runtime/BUILD

# Utility rule file for libomp-test-deps.

# Include the progress variables for this target.
include src/CMakeFiles/libomp-test-deps.dir/progress.make

src/CMakeFiles/libomp-test-deps: src/test-deps/.success


src/test-deps/.success: src/libomp.so
src/test-deps/.success: ../tools/check-depends.pl
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/lc432959/repos/hpc/llvm-openmp/runtime/BUILD/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating test-deps/.success"
	cd /home/lc432959/repos/hpc/llvm-openmp/runtime/BUILD/src && /usr/bin/cmake -E make_directory /home/lc432959/repos/hpc/llvm-openmp/runtime/BUILD/src/test-deps
	cd /home/lc432959/repos/hpc/llvm-openmp/runtime/BUILD/src && /usr/local_rwth/bin/perl /home/lc432959/repos/hpc/llvm-openmp/runtime/tools/check-depends.pl --os=lin --arch=32e --expected="libdl.so.2,libgcc_s.so.1,libc.so.6,ld-linux-x86-64.so.2,libpthread.so.0,libstdc++.so.6" /home/lc432959/repos/hpc/llvm-openmp/runtime/BUILD/src/libomp.so
	cd /home/lc432959/repos/hpc/llvm-openmp/runtime/BUILD/src && /usr/bin/cmake -E touch test-deps/.success

libomp-test-deps: src/CMakeFiles/libomp-test-deps
libomp-test-deps: src/test-deps/.success
libomp-test-deps: src/CMakeFiles/libomp-test-deps.dir/build.make

.PHONY : libomp-test-deps

# Rule to build all files generated by this target.
src/CMakeFiles/libomp-test-deps.dir/build: libomp-test-deps

.PHONY : src/CMakeFiles/libomp-test-deps.dir/build

src/CMakeFiles/libomp-test-deps.dir/clean:
	cd /home/lc432959/repos/hpc/llvm-openmp/runtime/BUILD/src && $(CMAKE_COMMAND) -P CMakeFiles/libomp-test-deps.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/libomp-test-deps.dir/clean

src/CMakeFiles/libomp-test-deps.dir/depend:
	cd /home/lc432959/repos/hpc/llvm-openmp/runtime/BUILD && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lc432959/repos/hpc/llvm-openmp/runtime /home/lc432959/repos/hpc/llvm-openmp/runtime/src /home/lc432959/repos/hpc/llvm-openmp/runtime/BUILD /home/lc432959/repos/hpc/llvm-openmp/runtime/BUILD/src /home/lc432959/repos/hpc/llvm-openmp/runtime/BUILD/src/CMakeFiles/libomp-test-deps.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/libomp-test-deps.dir/depend

