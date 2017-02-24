// RUN: %libomp-compile && env OMP_CANCELLATION=true %libomp-run | %sort-threads | FileCheck %s
// REQUIRES: ompt
#include "callback.h"
#include "omp.h"

int main()
{
  #pragma omp parallel num_threads(2)
  {
    if(omp_get_thread_num() == 0)
    {
      #pragma omp cancel parallel
    }
    else
    {
      sleep(1);
      #pragma omp cancellation point parallel
    }
  }


  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_task_create'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_cancel'

  // CHECK: {{^}}0: NULL_POINTER=[[NULL:.*$]]
  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_task_create: parent_task_id=[[PARENT_TASK_ID:[0-9]+]], parent_task_frame.exit=[[NULL]], parent_task_frame.reenter=[[NULL]], new_task_id=[[TASK_ID:[0-9]+]], parallel_function={{0x[0-f]*}}, task_type=ompt_task_initial=1, has_dependences=no
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_cancel: task_data=[[TASK_ID:[0-9]+]], flags=ompt_cancel_parallel|ompt_cancel_activated=17, codeptr_ra={{0x[0-f]*}}
  // CHECK: {{^}}[[OTHER_THREAD_ID:[0-9]+]]: ompt_event_cancel: task_data=[[TASK_ID:[0-9]+]], flags=ompt_cancel_parallel|ompt_cancel_detected=33, codeptr_ra={{0x[0-f]*}}

  return 0;
}
