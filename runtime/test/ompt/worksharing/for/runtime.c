// RUN: %libomp-compile-and-run | %sort-threads | %filecheck %S/base.h
// REQUIRES: ompt

#define SCHEDULE runtime
#include "base.h"
