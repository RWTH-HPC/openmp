// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %s
// REQUIRES: ompt
#include "callback.h"
#include <omp.h> 

int main()
{
  //initial task
  int task_type;
  ompt_get_task_info(0, &task_type, NULL, NULL, NULL, NULL);
  printf("%" PRIu64 ": ompt_event_task_create: task_type=%s=%d\n", ompt_get_thread_data()->value, "ompt_task_initial", task_type);

  int x;

  //implicit task
  #pragma omp parallel num_threads(1)
  {
    x++;
    ompt_get_task_info(0, &task_type, NULL, NULL, NULL, NULL);
    printf("%" PRIu64 ": ompt_event_task_create: task_type=%s=%d\n", ompt_get_thread_data()->value, "ompt_task_implicit|ompt_task_undeferred|ompt_task_untied", task_type);
  }

  #pragma omp parallel num_threads(1)
  {
    //explicit task
    #pragma omp task
    {
      x++;
    }

    //explicit task with undeferred
    #pragma omp task if(0)
    {
      x++;
    }

    //TODO:not working
    //explicit task with untied
    #pragma omp task untied
    {
      x++;
    }

    //TODO:not working
    //explicit task with final
    #pragma omp task final(1)
    {

      x++;
      //nested explicit task with final and deferred
      #pragma omp task
      {
        x++;
      }
    }

    //TODO:not working
    //explicit task with mergeable
    #pragma omp task mergeable
    {
      x++;
    }

    //TODO: merged task
  }






  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_task_create'


  // CHECK: {{^}}0: NULL_POINTER=[[NULL:.*$]]
  
  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_task_create: task_type=ompt_task_initial=1
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_create: task_type=ompt_task_implicit|ompt_task_undeferred|ompt_task_untied=402653186
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_create: parent_task_id={{[0-9]+}}, parent_task_frame.exit={{0x[0-f]+}}, parent_task_frame.reenter={{0x[0-f]+}}, new_task_id={{[0-9]+}}, parallel_function={{0x[0-f]+}}, task_type=ompt_task_explicit=4, has_dependences=no
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_create: parent_task_id={{[0-9]+}}, parent_task_frame.exit={{0x[0-f]+}}, parent_task_frame.reenter={{0x[0-f]+}}, new_task_id={{[0-9]+}}, parallel_function={{0x[0-f]+}}, task_type=ompt_task_explicit|ompt_task_undeferred|ompt_task_untied=402653188, has_dependences=no
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_create: parent_task_id={{[0-9]+}}, parent_task_frame.exit={{0x[0-f]+}}, parent_task_frame.reenter={{0x[0-f]+}}, new_task_id={{[0-9]+}}, parallel_function={{0x[0-f]+}}, task_type=ompt_task_explicit=4, has_dependences=no
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_create: parent_task_id={{[0-9]+}}, parent_task_frame.exit={{0x[0-f]+}}, parent_task_frame.reenter={{0x[0-f]+}}, new_task_id={{[0-9]+}}, parallel_function={{0x[0-f]+}}, task_type=ompt_task_explicit=4, has_dependences=no
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_create: parent_task_id={{[0-9]+}}, parent_task_frame.exit={{0x[0-f]+}}, parent_task_frame.reenter={{0x[0-f]+}}, new_task_id={{[0-9]+}}, parallel_function={{0x[0-f]+}}, task_type=ompt_task_explicit=4, has_dependences=no
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_create: parent_task_id={{[0-9]+}}, parent_task_frame.exit={{0x[0-f]+}}, parent_task_frame.reenter={{0x[0-f]+}}, new_task_id={{[0-9]+}}, parallel_function={{0x[0-f]+}}, task_type=ompt_task_explicit=4, has_dependences=no


  return 0;
}
