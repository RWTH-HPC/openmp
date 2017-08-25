// RUN: %libomp-compile-and-run | %sort-threads | %filecheck %S/base_split.h
// REQUIRES: ompt

#define SCHEDULE runtime
#include "base_split.h"
