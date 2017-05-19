# LLVM-openmp branch towards_tr4

LLVM/OpenMP runtime with changes towards TR4 compliance regarding OMPT

# Reference tool
For reference of the currently implemented OMPT API please check [callback.h](runtime/test/ompt/callback.h),
the tests (see below) are run with every push.

# how to configure/build:
## Make all implemented callbacks active:
    mkdir BUILD && cd BUILD
    cmake ../ -DLIBOMP_OMPT_SUPPORT=on
    make

## Make only minimal set of mandatory callbacks active:
    mkdir BUILD && cd BUILD
    cmake ../ -DLIBOMP_OMPT_SUPPORT=on -DLIBOMP_OMPT_OPTIONAL=off
    make

## Build & execute tests
The test tools of LLVM are needed, configure how to find them (these are built during LLVM build, but not installed):

    cmake . -DLIBOMP_LLVM_LIT_EXECUTABLE=/path/to/lit -DFILECHECK_EXECUTABLE=/path/to/FileCheck
    make check-libomp

# known issues of this branch:
* for problems with frame addresses compile the application with -fno-omit-frame-pointer
* not the whole interface is up-to-date yet. Inspect runtime/test/ompt/callback.h and runtime/src/include/45/ompt.h for reference
  * state information follows the old spec (TR2)
* gomp interface does not pass all tests

* Cancellation is broken in Clang compiler < 4.0. Deadlocks!
