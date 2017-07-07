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
          OMPT_WAIT(condition,1);
          #pragma omp atomic
          x++;
          OMPT_SIGNAL(condition);
        }
        #pragma omp task
        {
          OMPT_SIGNAL(condition);
          #pragma omp atomic
          x++;
          OMPT_WAIT(condition,2);
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

  return 0;
}
