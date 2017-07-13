// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %s
// REQUIRES: ompt, master_callback
#include "callback.h"
#include <omp.h>

int main()
{
  omp_lock_t lock;
  omp_init_lock(&lock);

  omp_test_lock(&lock);
  omp_unset_lock(&lock);

  omp_set_lock(&lock);
  omp_test_lock(&lock);
  omp_unset_lock(&lock);

  omp_destroy_lock(&lock);

  
  omp_nest_lock_t nest_lock;
  omp_init_nest_lock(&nest_lock);

  omp_test_nest_lock(&nest_lock);
  omp_unset_nest_lock(&nest_lock);

  omp_set_nest_lock(&nest_lock);
  omp_test_nest_lock(&nest_lock);
  omp_unset_nest_lock(&nest_lock);
  omp_unset_nest_lock(&nest_lock);

  int condition = 0;
  #pragma omp parallel num_threads(2)
  {
/*    #pragma omp master
    #pragma omp task
    {
      OMPT_WAIT(condition,1);
      omp_set_nest_lock(&nest_lock);
      OMPT_SIGNAL(condition);
      OMPT_WAIT(condition,3);
      omp_unset_nest_lock(&nest_lock);
    }
    #pragma omp master
    #pragma omp task
    {
      OMPT_SIGNAL(condition);
      OMPT_WAIT(condition,2);
      omp_test_nest_lock(&nest_lock); //should fail
      OMPT_SIGNAL(condition);
    }*/

    #pragma omp master
      omp_set_nest_lock(&nest_lock);
    #pragma omp barrier
    omp_test_nest_lock(&nest_lock); //should fail for non-master
    #pragma omp barrier
    #pragma omp master
    {
      omp_unset_nest_lock(&nest_lock);
      omp_unset_nest_lock(&nest_lock);
    }


  }

  omp_destroy_nest_lock(&nest_lock);

  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_mutex_acquire'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_mutex_acquired'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_mutex_released'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_nest_lock'

  // CHECK: 0: NULL_POINTER=[[NULL:.*$]]

  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_init_lock: wait_id=[[WAIT_ID:[0-9]+]], hint=0, impl={{[0-9]+}}, codeptr_ra={{0x[0-f]+}} 
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_wait_lock: wait_id=[[WAIT_ID]], hint=0, impl={{[0-9]+}}, codeptr_ra={{0x[0-f]+}}  
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_acquired_lock: wait_id=[[WAIT_ID]], codeptr_ra={{0x[0-f]+}}  
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_release_lock: wait_id=[[WAIT_ID]], codeptr_ra={{0x[0-f]+}}  
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_wait_lock: wait_id=[[WAIT_ID]], hint=0, impl={{[0-9]+}}, codeptr_ra={{0x[0-f]+}}  
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_acquired_lock: wait_id=[[WAIT_ID]], codeptr_ra={{0x[0-f]+}}  
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_wait_lock: wait_id=[[WAIT_ID]], hint=0, impl={{[0-9]+}}, codeptr_ra={{0x[0-f]+}}  
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_release_lock: wait_id=[[WAIT_ID]], codeptr_ra={{0x[0-f]+}}  
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_destroy_lock: wait_id=[[WAIT_ID]], codeptr_ra={{0x[0-f]+}}  

  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_init_nest_lock: wait_id=[[WAIT_ID:[0-9]+]], hint=0, impl={{[0-9]+}}, codeptr_ra={{0x[0-f]+}}  
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_wait_nest_lock: wait_id=[[WAIT_ID]], hint=0, impl={{[0-9]+}}, codeptr_ra=[[NULL]] 
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_acquired_nest_lock_first: wait_id=[[WAIT_ID]], codeptr_ra=[[NULL]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_release_nest_lock_last: wait_id=[[WAIT_ID]], codeptr_ra={{0x[0-f]+}}  
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_wait_nest_lock: wait_id=[[WAIT_ID]], hint=0, impl={{[0-9]+}}, codeptr_ra={{0x[0-f]+}}  
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_acquired_nest_lock_first: wait_id=[[WAIT_ID]], codeptr_ra={{0x[0-f]+}}  
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_wait_nest_lock: wait_id=[[WAIT_ID]], hint=0, impl={{[0-9]+}}, codeptr_ra=[[NULL]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_acquired_nest_lock_next: wait_id=[[WAIT_ID]], codeptr_ra=[[NULL]] 
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_release_nest_lock_prev: wait_id=[[WAIT_ID]], codeptr_ra={{0x[0-f]+}}  
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_release_nest_lock_last: wait_id=[[WAIT_ID]], codeptr_ra={{0x[0-f]+}}  


  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_wait_nest_lock: wait_id=[[WAIT_ID]], hint=0
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_acquired_nest_lock_first: wait_id=[[WAIT_ID]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_wait_nest_lock: wait_id=[[WAIT_ID]], hint=0
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_acquired_nest_lock_next: wait_id=[[WAIT_ID]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_release_nest_lock_prev: wait_id=[[WAIT_ID]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_release_nest_lock_last: wait_id=[[WAIT_ID]]

  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_destroy_nest_lock: wait_id=[[WAIT_ID]]


  // CHECK: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_wait_nest_lock: wait_id=[[WAIT_ID]], hint=0
  // ___CHECK: {{^}}[[THREAD_ID]]: ompt_event_acquired_nest_lock_first: wait_id=[[WAIT_ID]], codeptr_ra={{0x[0-f]+}}   
  // ___CHECK: {{^}}[[THREAD_ID]]: ompt_event_release_nest_lock_last: wait_id=[[WAIT_ID]], codeptr_ra={{0x[0-f]+}}   


  return 0;
}
