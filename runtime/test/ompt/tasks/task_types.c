// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %s
// REQUIRES: ompt
#include "callback.h"
#include <omp.h>

void print_task_type()
{
  #pragma omp critical
  {
    int task_type;
    char buffer[2048];
    ompt_get_task_info(0, &task_type, NULL, NULL, NULL, NULL);
    format_task_type(task_type, buffer);
    printf("%" PRIu64 ": task_type=%s=%d\n", ompt_get_thread_data()->value, buffer, task_type);
  }
};

int main()
{
  //initial task
  print_task_type();

  int x;
  //implicit task
  #pragma omp parallel num_threads(1)
  {
    print_task_type();
    x++;
  }

  #pragma omp parallel num_threads(1)
  {
    //explicit task
    #pragma omp task
    {
      print_task_type();
      x++;
    }

    //explicit task with undeferred
    #pragma omp task if(0)
    {
      print_task_type();
      x++;
    }

/*    //TODO:not working
    //explicit task with untied
    #pragma omp task untied
    {
      print_task_type();
      x++;
    }*/

    //TODO:not working
    //explicit task with final
    #pragma omp task final(1)
    {
      print_task_type();
      x++;
      //nested explicit task with final and deferred
      #pragma omp task
      {
        print_task_type();
        x++;
      }
    }

    //TODO:not working
    //explicit task with mergeable
    #pragma omp task mergeable
    {
      print_task_type();
      x++;
    }

    //TODO: merged task
  }






  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_task_create'


  // CHECK: {{^}}0: NULL_POINTER=[[NULL:.*$]]
  
  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: task_type=ompt_task_initial=1
  // CHECK: {{^}}[[MASTER_ID]]: task_type=ompt_task_implicit|ompt_task_undeferred=134217730
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_create: parent_task_id={{[0-9]+}}, parent_task_frame.exit={{0x[0-f]+}}, parent_task_frame.reenter={{0x[0-f]+}}, new_task_id={{[0-9]+}}, parallel_function={{0x[0-f]+}}, task_type=ompt_task_explicit=4, has_dependences=no
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_create: parent_task_id={{[0-9]+}}, parent_task_frame.exit={{0x[0-f]+}}, parent_task_frame.reenter={{0x[0-f]+}}, new_task_id={{[0-9]+}}, parallel_function={{0x[0-f]+}}, task_type=ompt_task_explicit|ompt_task_undeferred=134217732, has_dependences=no
  // ____CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_create: parent_task_id={{[0-9]+}}, parent_task_frame.exit={{0x[0-f]+}}, parent_task_frame.reenter={{0x[0-f]+}}, new_task_id={{[0-9]+}}, parallel_function={{0x[0-f]+}}, task_type=ompt_task_explicit=4, has_dependences=no
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_create: parent_task_id={{[0-9]+}}, parent_task_frame.exit={{0x[0-f]+}}, parent_task_frame.reenter={{0x[0-f]+}}, new_task_id={{[0-9]+}}, parallel_function={{0x[0-f]+}}, task_type=ompt_task_explicit=4, has_dependences=no
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_create: parent_task_id={{[0-9]+}}, parent_task_frame.exit={{0x[0-f]+}}, parent_task_frame.reenter={{0x[0-f]+}}, new_task_id={{[0-9]+}}, parallel_function={{0x[0-f]+}}, task_type=ompt_task_explicit=4, has_dependences=no
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_create: parent_task_id={{[0-9]+}}, parent_task_frame.exit={{0x[0-f]+}}, parent_task_frame.reenter={{0x[0-f]+}}, new_task_id={{[0-9]+}}, parallel_function={{0x[0-f]+}}, task_type=ompt_task_explicit=4, has_dependences=no


  return 0;
}
