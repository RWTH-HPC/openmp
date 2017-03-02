// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %s
// REQUIRES: ompt
#include "callback.h"
#include <omp.h>   
#include <unistd.h>

int main()
{
  #pragma omp parallel num_threads(2)
  {
    #pragma omp master
    {
      #pragma omp task
      {
        sleep(1);
        #pragma omp taskyield
        sleep(1);
      }
      #pragma omp task
      {
        sleep(1);
      }
    }
  }


  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_task_create'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_task_schedule'

  
  // CHECK: {{^}}0: NULL_POINTER=[[NULL:.*$]]
  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_task_create: parent_task_id={{[0-9]+}}, parent_task_frame.exit=[[NULL]], parent_task_frame.reenter=[[NULL]], new_task_id=[[PARENT_TASK:[0-9]+]], parallel_function=0x{{[0-f]+}}, task_type=ompt_task_initial=1, has_dependences=no
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_create: parent_task_id=[[PARENT_TASK]], parent_task_frame.exit={{0x[0-f]+}}, parent_task_frame.reenter={{0x[0-f]+}}, new_task_id=[[TASK_ONE:[0-9]+]], parallel_function={{0x[0-f]+}}, task_type=ompt_task_explicit=3, has_dependences=no
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_create: parent_task_id=[[PARENT_TASK]], parent_task_frame.exit={{0x[0-f]+}}, parent_task_frame.reenter={{0x[0-f]+}}, new_task_id=[[TASK_TWO:[0-9]+]], parallel_function={{0x[0-f]+}}, task_type=ompt_task_explicit=3, has_dependences=no


  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_schedule: first_task_id={{[0-9]+}}, second_task_id=[[TASK_TWO]], prior_task_status=ompt_task_others=4
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_schedule: first_task_id=[[TASK_TWO]], second_task_id={{[0-9]+}}, prior_task_status=ompt_task_complete=1


  // CHECK: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_task_schedule: first_task_id={{[0-9]+}}, second_task_id=[[TASK_ONE]], prior_task_status=ompt_task_others=4
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_task_schedule: first_task_id=[[TASK_ONE]], second_task_id=[[TASK_ONE]], prior_task_status=ompt_task_yield=2
  // missing schedule callbacks?
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_task_schedule: first_task_id=[[TASK_ONE]], second_task_id={{[0-9]+}}, prior_task_status=ompt_task_complete=1





  return 0;
}
