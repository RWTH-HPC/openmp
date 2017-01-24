# LLVM-openmp branch towards_tr4

LLVM/OpenMP runtime with changes towards TR4 compliance regarding OMPT

# how to configure/build:
## Make all implemented callbacks active:
    mkdir BUILD && cd BUILD
    cmake ../ -DLIBOMP_OMPT_SUPPORT=on
    make

## Make only minimal set of mandatory callbacks active:
    mkdir BUILD && cd BUILD
    cmake ../ -DLIBOMP_OMPT_SUPPORT=on -DLIBOMP_OMPT_BLAME=off -DLIBOMP_OMPT_TRACE=off
    make

## Build & execute tests
The test tools of LLVM are needed, configure how to find them (these are built during LLVM build, but not installed):

    cmake . -DLIBOMP_LLVM_LIT_EXECUTABLE=/path/to/lit -DFILECHECK_EXECUTABLE=/path/to/FileCheck
    make check-libomp

# known issues of this branch:
* codeptr_ra argument only works if runtime is compiled with optimization and the kmp interface is used.
* for problems with frame addresses compile the application with -fno-omit-frame-pointer
* not the whole interface is up-to-date yet. Inspect runtime/test/ompt/callback.h and runtime/src/include/45/ompt.h for reference
  * callbacks for flush/cancel/yield .. are missing
  * some API functions follow the old spec (TR2)
  * state information follows the old spec (TR2)
* ompt.h header file is only updated for LIBOMP_OMP_VERSION=45
* gomp interface is not tested yet
