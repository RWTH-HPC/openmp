// RUN:  %libomp-compile-and-run | %sort-threads | FileCheck %s
// REQUIRES: ompt, cancel, taskgroup

#include "callback.h"
#include <unistd.h>  
#include <stdio.h>

int main()
{
  int condition=0;
  int x=0;
  #pragma omp parallel num_threads(2)
  {
    #pragma omp master
    {
      #pragma omp taskgroup
      {
        #pragma omp task
        {
          #pragma omp atomic
          x++;
        }
      }
    }
  }


  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_master'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_task_create'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_task_schedule'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_cancel'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_thread_begin'


  // CHECK: {{^}}0: NULL_POINTER=[[NULL:.*$]]

  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_taskgroup_begin: parallel_id=[[PARALLEL_ID:[0-9]+]], task_id=[[TASK_ID:[0-9]+]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_wait_taskgroup_begin: parallel_id=[[PARALLEL_ID]], task_id=[[TASK_ID]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_wait_taskgroup_end: parallel_id=[[PARALLEL_ID]], task_id=[[TASK_ID]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_taskgroup_end: parallel_id=[[PARALLEL_ID]], task_id=[[TASK_ID]]


  return 0;
}
