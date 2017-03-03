// RUN: %libomp-compile-and-run | FileCheck %s
// REQUIRES: ompt
#include "callback.h"
#include <omp.h>

int main()
{
  #pragma omp parallel num_threads(1)
  {
  	printf("%" PRIu64 ": omp_get_num_places()=%d\n", ompt_get_thread_data()->value, omp_get_num_places());
  	printf("%" PRIu64 ": ompt_get_num_places()=%d\n", ompt_get_thread_data()->value, ompt_get_num_places());

  	printf("%" PRIu64 ": omp_get_place_num()=%d\n", ompt_get_thread_data()->value, omp_get_place_num());
  	printf("%" PRIu64 ": ompt_get_place_num()=%d\n", ompt_get_thread_data()->value, ompt_get_place_num());

  }

  // Check if libomp supports the callbacks for this test.

  // CHECK: 0: NULL_POINTER=[[NULL:.*$]]

  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: omp_get_num_places()=[[NUM_PLACES:[0-9]+]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_get_num_places()=[[NUM_PLACES]]
  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: omp_get_place_num()=[[PLACE_NUM:[-]?[0-9]+]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_get_place_num()=[[PLACE_NUM]]

  return 0;
}
