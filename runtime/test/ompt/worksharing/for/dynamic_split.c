// RUN: %libomp-compile-and-run | %sort-threads | %filecheck %S/base_split.h
// RUN: %libomp-compile-and-run | %sort-threads | %filecheck-custom --check-prefix=CHECK-LOOP %S/base_split.h
// REQUIRES: ompt

#define SCHEDULE dynamic
#include "base_split.h"
