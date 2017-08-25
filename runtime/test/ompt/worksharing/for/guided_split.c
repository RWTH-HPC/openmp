// RUN: %libomp-compile-and-run | %sort-threads | %filecheck %S/base_split.h
// REQUIRES: ompt

#define SCHEDULE guided
#include "base_split.h"
