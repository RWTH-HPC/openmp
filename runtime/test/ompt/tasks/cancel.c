// RUN: %libomp-compile && env OMP_CANCELLATION=true %libomp-run | %sort-threads | FileCheck %s
// REQUIRES: ompt
#include "callback.h"
#include <omp.h>   

int main()
{
  //omp_set_cancellation(1);
  #pragma omp taskgroup
  {
    int x = 0;
    int i;
    for(i = 0; i < 2; i++)
    {
      #pragma omp task shared(x)
      {
        //#pragma omp cancellation point taskgroup
        x++;
        if(x == 1)
        {
          #pragma omp cancel taskgroup
        }
      }
    }
  }


  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_cancel'

  // CHECK: {{^}}0: NULL_POINTER=[[NULL:.*$]]
  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_task_create: parent_task_id=[[PARENT_TASK_ID:[0-9]+]], parent_task_frame.exit=[[NULL]], parent_task_frame.reenter={{0x[0-f]*}}, new_task_id=[[TASK_ID:[0-9]+]], parallel_function={{0x[0-f]*}}, task_type=ompt_task_explicit=3, has_dependences=no
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_cancel: task_data=[[TASK_ID:[0-9]+]], flags=72, codeptr_ra={{0x[0-f]*}}

  return 0;
}
