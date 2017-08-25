// RUN: %libomp-compile-and-run | %sort-threads | %filecheck %S/base_split.h
// REQUIRES: ompt
// GCC doesn't call runtime for static schedule
// XFAIL: gcc

#define SCHEDULE static
#include "base_split.h"
