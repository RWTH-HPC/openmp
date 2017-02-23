// RUN:  %libomp-compile && env OMP_CANCELLATION=true %libomp-run | %sort-threads | FileCheck %s
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
    #pragma omp master
    {
      #pragma omp taskgroup
      {
        #pragma omp task
        {
          sleep(2);
          #pragma omp cancellation point taskgroup
        }
        #pragma omp task
        {
          sleep(2);
          #pragma omp cancellation point taskgroup
        }
        #pragma omp task
        {
          sleep(2);
          #pragma omp cancellation point taskgroup
        }
        sleep(1);
        #pragma omp task if(0)
        {
          print_ids(0);
          #pragma omp cancel taskgroup
        }
      }
    }
    #pragma omp barrier
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
  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_master_begin: parallel_id=[[PARALLEL_ID:[0-9]+]], task_id=[[PARENT_TASK_ID:[0-9]+]], codeptr_ra={{0x[0-f]*}}

  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_create: parent_task_id=[[PARENT_TASK_ID]], parent_task_frame.exit={{0x[0-f]*}}, parent_task_frame.reenter={{0x[0-f]*}}, new_task_id=[[FIRST_TASK_ID:[0-9]+]], parallel_function={{0x[0-f]*}}, task_type=ompt_task_explicit=3, has_dependences=no
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_create: parent_task_id=[[PARENT_TASK_ID]], parent_task_frame.exit={{0x[0-f]*}}, parent_task_frame.reenter={{0x[0-f]*}}, new_task_id=[[SECOND_TASK_ID:[0-9]+]], parallel_function={{0x[0-f]*}}, task_type=ompt_task_explicit=3, has_dependences=no
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_create: parent_task_id=[[PARENT_TASK_ID]], parent_task_frame.exit={{0x[0-f]*}}, parent_task_frame.reenter={{0x[0-f]*}}, new_task_id=[[THIRD_TASK_ID:[0-9]+]], parallel_function={{0x[0-f]*}}, task_type=ompt_task_explicit=3, has_dependences=no

  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_create: parent_task_id=[[PARENT_TASK_ID]], parent_task_frame.exit={{0x[0-f]*}}, parent_task_frame.reenter={{0x[0-f]*}}, new_task_id=[[CANCEL_TASK_ID:[0-9]+]], parallel_function={{0x[0-f]*}}, task_type=ompt_task_explicit=3, has_dependences=no
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_schedule: first_task_id=[[PARENT_TASK_ID]], second_task_id=[[CANCEL_TASK_ID]], prior_task_status=4

  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_cancel: task_data=[[CANCEL_TASK_ID]], flags=ompt_cancel_taskgroup|ompt_cancel_activated=24, codeptr_ra={{0x[0-f]*}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_cancel: task_data=[[THIRD_TASK_ID]], flags=ompt_cancel_taskgroup|ompt_cancel_discarded_task=72, codeptr_ra={{0x[0-f]*}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_cancel: task_data=[[SECOND_TASK_ID]], flags=ompt_cancel_taskgroup|ompt_cancel_discarded_task=72, codeptr_ra={{0x[0-f]*}}
  
  // CHECK: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_thread_begin: thread_type=ompt_thread_worker=2, thread_id=[[THREAD_ID]]
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_cancel: task_data=[[FIRST_TASK_ID]], flags=ompt_cancel_taskgroup|ompt_cancel_detected=40, codeptr_ra={{0x[0-f]*}}

  return 0;
}
