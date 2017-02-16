// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %s
// REQUIRES: ompt
#include "callback.h"
#include <omp.h> 
#include <unistd.h>  

int main()
{
  omp_set_nested(0);
  print_frame(0);
  #pragma omp parallel num_threads(2)
  {
    print_frame(1);
    print_ids(0);
    print_ids(1);
    print_frame(0);
    #pragma omp master
    {
      print_ids(0);
      #pragma omp task
      {
        print_frame(1);
        print_ids(0);
        print_ids(1);
        print_ids(2);
      }
      sleep(1);
      print_ids(0);
    }
    #pragma omp barrier
    print_ids(0);
  }


  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_event_parallel_begin'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_event_parallel_end'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_event_implicit_task_begin'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_event_implicit_task_end'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_event_barrier_begin'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_event_barrier_end'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_event_wait_barrier_begin'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_event_wait_barrier_end'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_event_task_begin'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_event_task_switch'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_event_task_end'


  // CHECK: {{^}}0: NULL_POINTER=[[NULL:.*$]]
  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: __builtin_frame_address(0)=[[MAIN_REENTER:0x[0-f]+]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_parallel_begin: parent_task_id=[[PARENT_TASK_ID:[0-9]+]], parent_task_frame.exit=[[NULL]], parent_task_frame.reenter=[[MAIN_REENTER]], parallel_id=[[PARALLEL_ID:[0-9]+]], requested_team_size=2, parallel_function=0x{{[0-f]+}}, invoker=[[PARALLEL_INVOKER:[0-9]+]]
  // nested parallel masters
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID:[0-9]+]]
  // CHECK: {{^}}[[MASTER_ID]]: __builtin_frame_address(1)=[[EXIT:0x[0-f]+]]
  // CHECK: {{^}}[[MASTER_ID]]: task level 0: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]], exit_frame=[[EXIT]], reenter_frame=[[NULL]]
  // CHECK: {{^}}[[MASTER_ID]]: task level 1: parallel_id=0, task_id=[[PARENT_TASK_ID]], exit_frame=[[NULL]], reenter_frame=[[MAIN_REENTER]]
  // CHECK: {{^}}[[MASTER_ID]]: __builtin_frame_address(0)=[[REENTER:0x[0-f]+]]
  // CHECK: {{^}}[[MASTER_ID]]: task level 0: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]], exit_frame=[[EXIT]], reenter_frame=[[NULL]]
  // <- ompt_event_task_create would be expected here
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_create: parent_task_id=[[IMPLICIT_TASK_ID]], parent_task_frame.exit=[[EXIT]], parent_task_frame.reenter=[[REENTER]], new_task_id=[[TASK_ID:[0-9]+]], parallel_function=[[TASK_FUNCTION:0x[0-f]+]]
  // CHECK: {{^}}[[MASTER_ID]]: task level 0: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]], exit_frame=[[EXIT]], reenter_frame=[[NULL]]
  // explicit barrier after master
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_barrier_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // CHECK: {{^}}[[MASTER_ID]]: task level 0: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]], exit_frame=[[EXIT]], reenter_frame=[[REENTER]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_barrier_end: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // implicit barrier parallel
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_barrier_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // CHECK: {{^}}[[MASTER_ID]]: task level 0: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]], exit_frame=[[NULL]], reenter_frame=[[NULL]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_barrier_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]

  // CHECK: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_implicit_task_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID:[0-9]+]]
  // CHECK: {{^}}[[THREAD_ID]]: __builtin_frame_address(1)=[[EXIT:0x[0-f]+]]
  // CHECK: {{^}}[[THREAD_ID]]: task level 0: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]], exit_frame=[[EXIT]], reenter_frame=[[NULL]]
  // CHECK: {{^}}[[THREAD_ID]]: task level 1: parallel_id=0, task_id=[[PARENT_TASK_ID]], exit_frame=[[NULL]], reenter_frame=[[MAIN_REENTER]]
  // CHECK: {{^}}[[THREAD_ID]]: __builtin_frame_address(0)=[[REENTER:0x[0-f]+]]
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_barrier_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // CHECK: {{^}}[[THREAD_ID]]: task level 0: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]], exit_frame=[[EXIT]], reenter_frame=[[REENTER]]
  // this is expected to come earlier and at MASTER:
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_task_schedule: first_task_id=[[IMPLICIT_TASK_ID]], second_task_id=[[TASK_ID]]
  // CHECK: {{^}}[[THREAD_ID]]: __builtin_frame_address(1)=[[TASK_EXIT:0x[0-f]+]]
  // CHECK: {{^}}[[THREAD_ID]]: task level 0: parallel_id=[[PARALLEL_ID]], task_id=[[TASK_ID]], exit_frame=[[TASK_EXIT]], reenter_frame=[[NULL]]
  // CHECK: {{^}}[[THREAD_ID]]: task level 1: parallel_id=0, task_id=[[IMPLICIT_TASK_ID]], exit_frame=[[EXIT]], reenter_frame=[[REENTER]]
  // CHECK: {{^}}[[THREAD_ID]]: task level 2: parallel_id=0, task_id=[[PARENT_TASK_ID]], exit_frame=[[NULL]], reenter_frame=[[MAIN_REENTER]]
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_task_schedule: first_task_id=[[TASK_ID]], second_task_id=[[IMPLICIT_TASK_ID]]
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_task_end: task_id=[[TASK_ID]]
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_barrier_end: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_barrier_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // CHECK: {{^}}[[THREAD_ID]]: task level 0: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]], exit_frame=[[NULL]], reenter_frame=[[NULL]]
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_barrier_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]



  return 0;
}
