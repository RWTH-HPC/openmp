// RUN: %libomp-compile-and-run | FileCheck %s
// REQUIRES: ompt
#include "callback.h"
#include <omp.h>

int main()
{
  omp_set_nested(1);

  printf("%d\n", omp_get_max_active_levels());
  printf("%d\n", omp_get_thread_limit());
  printf("%d\n", omp_get_dynamic());
  omp_set_max_active_levels(1);
  printf("%d\n", omp_get_max_active_levels());

  #pragma omp parallel num_threads(1)
  {
    print_ids(0);
    print_ids(1);
    #pragma omp parallel num_threads(1)
    {
      print_ids(0);
      print_ids(1);
      print_ids(2);
    }
  }

  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_parallel_begin'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_parallel_end'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_implicit_task'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_mutex_acquire'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_mutex_acquired'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_mutex_released'

  // CHECK: 0: NULL_POINTER=[[NULL:.*$]]
  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_parallel_begin: parent_task_id=[[PARENT_TASK_ID:[0-9]+]], parent_task_frame.exit=[[NULL]], parent_task_frame.reenter={{0x[0-f]+}}, parallel_id=[[PARALLEL_ID:[0-9]+]], requested_team_size=1, parallel_function=0x{{[0-f]+}}, invoker=[[PARALLEL_INVOKER:[0-9]+]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID:[0-9]+]]
  // CHECK: {{^}}[[MASTER_ID]]: task level 0: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // CHECK: {{^}}[[MASTER_ID]]: task level 1: parallel_id=0, task_id=[[PARENT_TASK_ID]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_parallel_begin: parent_task_id=[[IMPLICIT_TASK_ID]], parent_task_frame.exit={{0x[0-f]+}}, parent_task_frame.reenter={{0x[0-f]+}}, parallel_id=[[NESTED_PARALLEL_ID:[0-9]+]], requested_team_size=1, parallel_function=[[NESTED_PARALLEL_FUNCTION:0x[0-f]+]], invoker=[[PARALLEL_INVOKER]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_begin: parallel_id=[[NESTED_PARALLEL_ID]], task_id=[[NESTED_IMPLICIT_TASK_ID:[0-9]+]]
  // CHECK: {{^}}[[MASTER_ID]]: task level 0: parallel_id=[[NESTED_PARALLEL_ID]], task_id=[[NESTED_IMPLICIT_TASK_ID]]
  // CHECK: {{^}}[[MASTER_ID]]: task level 1: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // CHECK: {{^}}[[MASTER_ID]]: task level 2: parallel_id=0, task_id=[[PARENT_TASK_ID]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_end: parallel_id={{[0-9]+}}, task_id=[[NESTED_IMPLICIT_TASK_ID]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_parallel_end: parallel_id=[[NESTED_PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]], invoker=[[PARALLEL_INVOKER]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]
  return 0;
}
