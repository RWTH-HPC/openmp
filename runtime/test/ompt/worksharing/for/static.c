// RUN: %libomp-compile-and-run | %sort-threads | %filecheck %S/base.h
// REQUIRES: ompt
// GCC doesn't call runtime for static schedule
// XFAIL: gcc

#define SCHEDULE static
#include "base.h"
