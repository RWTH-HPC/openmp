// RUN: %libomp-compile && env OMP_CANCELLATION=true %libomp-run | %sort-threads | FileCheck %s
// REQUIRES: ompt
#include "callback.h"

int main()
{
  #pragma omp parallel num_threads(1)
  {
    int x = 0;
    int i;
    #pragma omp for
    for(i = 0; i < 2; i++)
    {
      #pragma omp cancel for
    }

    #pragma omp sections
    {
      #pragma omp section
      {
        #pragma omp cancel sections
      }
      #pragma omp section
      {
        #pragma omp cancel sections
      }
    }
  }


  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_cancel'

  // CHECK: {{^}}0: NULL_POINTER=[[NULL:.*$]]
  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_task_create: parent_task_id=[[PARENT_TASK_ID:[0-9]+]], parent_task_frame.exit=[[NULL]], parent_task_frame.reenter=[[NULL]], new_task_id=[[TASK_ID:[0-9]+]], parallel_function={{0x[0-f]*}}, task_type=ompt_task_initial=1, has_dependences=no
  
  // cancel for
  // ___CHECK: {{^}}[[MASTER_ID]]: ompt_event_cancel: task_data=[[TASK_ID:[0-9]+]], flags=20, codeptr_ra={{0x[0-f]*}}

  // cancel sections
  // ___CHECK: {{^}}[[MASTER_ID]]: ompt_event_cancel: task_data=[[TASK_ID:[0-9]+]], flags=18, codeptr_ra={{0x[0-f]*}}

  return 0;
}
