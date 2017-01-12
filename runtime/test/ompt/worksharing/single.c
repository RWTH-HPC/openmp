// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %s
// REQUIRES: ompt
#include "callback.h"
#include <omp.h>

int main()
{
  int x = 0;
  #pragma omp parallel num_threads(2)
  {
    #pragma omp single
    {
      x++;
    }
  }

  printf("x=%d\n", x);

  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_work'

  // CHECK: 0: NULL_POINTER=[[NULL:.*$]]

  // CHECK: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_single_in_block_begin: parallel_id=[[PARALLEL_ID:[0-9]+]], parent_task_id=[[TASK_ID:[0-9]+]], workshare_function={{0x[0-f]+}}, count=1
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_single_in_block_end: parallel_id=[[PARALLEL_ID]], task_id=[[TASK_ID]], workshare_function={{0x[0-f]+}}, count=1

  // CHECK: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_single_others_begin: parallel_id=[[PARALLEL_ID:[0-9]+]], task_id=[[TASK_ID:[0-9]+]], workshare_function={{0x[0-f]+}}, count=1
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_single_others_end: parallel_id=[[PARALLEL_ID]], task_id=[[TASK_ID]], workshare_function={{0x[0-f]+}}, count=1



  return 0;
}
