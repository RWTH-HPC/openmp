// RUN: %libomp-compile-and-run | %sort-threads | %filecheck %S/base.h
// REQUIRES: ompt
// GCC doesn't call runtime for auto = static schedule
// XFAIL: gcc

#define SCHEDULE auto
#include "base.h"
