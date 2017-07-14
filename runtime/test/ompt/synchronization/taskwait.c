// RUN: %libomp-compile-and-run | %sort-threads | %filecheck %s
// REQUIRES: ompt, master_callback
#include "callback.h"
#include <omp.h>

int main()
{
  int x = 0;
  #pragma omp parallel num_threads(2)
  {
    #pragma omp master
    #pragma omp task
    {
      x++;
    }

    #pragma omp taskwait
  }

  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_sync_region'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_sync_region_wait'

  // CHECK: 0: NULL_POINTER=[[NULL:.*$]]

  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_taskwait_begin: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, codeptr_ra=
  // CHECK-WAIT: {{^}}[[MASTER_ID]]: ompt_event_wait_taskwait_begin: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, codeptr_ra=
  // CHECK-WAIT: {{^}}[[MASTER_ID]]: ompt_event_wait_taskwait_end: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, codeptr_ra=
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_taskwait_end: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, codeptr_ra=


  // CHECK: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_taskwait_begin: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, codeptr_ra=
  // CHECK-WAIT: {{^}}[[THREAD_ID]]: ompt_event_wait_taskwait_begin: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, codeptr_ra=
  // CHECK-WAIT: {{^}}[[THREAD_ID]]: ompt_event_wait_taskwait_end: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, codeptr_ra=
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_taskwait_end: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, codeptr_ra=

  return 0;
}
