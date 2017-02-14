// RUN: %libomp-compile-and-run | FileCheck %s
// REQUIRES: ompt
#include "callback.h"
#include <omp.h>

int main()
{
  int x = 0;
  #pragma omp parallel num_threads(2)
  {
    #pragma omp master
    {
      x++;
    }
    print_current_address();
  }

  printf("%" PRIu64 ": x=%d\n", ompt_get_thread_data()->value, x);

  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_master'

  // CHECK: 0: NULL_POINTER=[[NULL:.*$]]

  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_master_begin: parallel_id=[[PARALLEL_ID:[0-9]+]], task_id=[[TASK_ID:[0-9]+]], codeptr_ra=[[RETURN_ADDRESS:0x[0-f]+]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_master_end: parallel_id=[[PARALLEL_ID]], task_id=[[TASK_ID]], codeptr_ra=[[RETURN_ADDRESS:0x[0-f]+]]
  // CHECK: {{^}}[[MASTER_ID]]: current_address=[[RETURN_ADDRESS]]


  return 0;
}
