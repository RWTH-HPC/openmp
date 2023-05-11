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

# Utility rule file for libomp-needed-headers.

# Include the progress variables for this target.
include src/CMakeFiles/libomp-needed-headers.dir/progress.make

src/CMakeFiles/libomp-needed-headers: src/kmp_i18n_id.inc
src/CMakeFiles/libomp-needed-headers: src/kmp_i18n_default.inc


src/kmp_i18n_id.inc: ../src/i18n/en_US.txt
src/kmp_i18n_id.inc: ../tools/message-converter.pl
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/lc432959/repos/hpc/llvm-openmp/runtime/BUILD/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating kmp_i18n_id.inc"
	cd /home/lc432959/repos/hpc/llvm-openmp/runtime/BUILD/src && /usr/local_rwth/bin/perl /home/lc432959/repos/hpc/llvm-openmp/runtime/tools/message-converter.pl --os=lin --prefix=kmp_i18n --enum=kmp_i18n_id.inc /home/lc432959/repos/hpc/llvm-openmp/runtime/src/i18n/en_US.txt

src/kmp_i18n_default.inc: ../src/i18n/en_US.txt
src/kmp_i18n_default.inc: ../tools/message-converter.pl
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/lc432959/repos/hpc/llvm-openmp/runtime/BUILD/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Generating kmp_i18n_default.inc"
	cd /home/lc432959/repos/hpc/llvm-openmp/runtime/BUILD/src && /usr/local_rwth/bin/perl /home/lc432959/repos/hpc/llvm-openmp/runtime/tools/message-converter.pl --os=lin --prefix=kmp_i18n --default=kmp_i18n_default.inc /home/lc432959/repos/hpc/llvm-openmp/runtime/src/i18n/en_US.txt

libomp-needed-headers: src/CMakeFiles/libomp-needed-headers
libomp-needed-headers: src/kmp_i18n_id.inc
libomp-needed-headers: src/kmp_i18n_default.inc
libomp-needed-headers: src/CMakeFiles/libomp-needed-headers.dir/build.make

.PHONY : libomp-needed-headers

# Rule to build all files generated by this target.
src/CMakeFiles/libomp-needed-headers.dir/build: libomp-needed-headers

.PHONY : src/CMakeFiles/libomp-needed-headers.dir/build

src/CMakeFiles/libomp-needed-headers.dir/clean:
	cd /home/lc432959/repos/hpc/llvm-openmp/runtime/BUILD/src && $(CMAKE_COMMAND) -P CMakeFiles/libomp-needed-headers.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/libomp-needed-headers.dir/clean

src/CMakeFiles/libomp-needed-headers.dir/depend:
	cd /home/lc432959/repos/hpc/llvm-openmp/runtime/BUILD && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lc432959/repos/hpc/llvm-openmp/runtime /home/lc432959/repos/hpc/llvm-openmp/runtime/src /home/lc432959/repos/hpc/llvm-openmp/runtime/BUILD /home/lc432959/repos/hpc/llvm-openmp/runtime/BUILD/src /home/lc432959/repos/hpc/llvm-openmp/runtime/BUILD/src/CMakeFiles/libomp-needed-headers.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/libomp-needed-headers.dir/depend
