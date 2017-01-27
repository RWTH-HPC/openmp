// RUN: %libomp-compile-and-run | FileCheck %s
// REQUIRES: ompt
#include "callback.h"
#include <omp.h>

int main()
{
  omp_lock_t lock;
  printf("%" PRIu64 ": &lock: %lli\n", ompt_get_thread_data()->value, &lock);
  omp_init_lock(&lock);
  omp_set_lock(&lock);
  omp_unset_lock(&lock);
  omp_destroy_lock(&lock);


  omp_nest_lock_t nest_lock;
  printf("%" PRIu64 ": &nest_lock: %lli\n", ompt_get_thread_data()->value, &nest_lock);
  omp_init_nest_lock(&nest_lock);
  omp_set_nest_lock(&nest_lock);
  omp_set_nest_lock(&nest_lock);
  omp_unset_nest_lock(&nest_lock);
  omp_unset_nest_lock(&nest_lock);
  omp_destroy_nest_lock(&nest_lock);

  //print_retadd();
  #pragma omp critical
  {
    print_ids(0);
  }


  int x = 3;
  #pragma omp atomic
  x++;


  #pragma omp ordered
  {
    print_ids(0);
  }


  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_mutex_acquire'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_mutex_acquired'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_mutex_released'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_nest_lock'

  // CHECK: 0: NULL_POINTER=[[NULL:.*$]]

  // TODO: check wait ids
  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: &lock: [[WAIT_ID:[0-9]+]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_init_lock: wait_id=[[WAIT_ID]], hint={{[0-9]+}}, impl={{[0-9]+}}, return_address={{0x[0-f]+}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_wait_lock: wait_id=[[WAIT_ID]], hint={{[0-9]+}}, impl={{[0-9]+}}, return_address={{0x[0-f]+}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_acquired_lock: wait_id=[[WAIT_ID]], return_address={{0x[0-f]+}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_release_lock: wait_id=[[WAIT_ID]], return_address={{0x[0-f]+}}

  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_init_nest_lock: wait_id=[[WAIT_ID:[0-9]+]], hint={{[0-9]+}}, impl={{[0-9]+}}, return_address={{0x[0-f]+}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_wait_nest_lock: wait_id=[[WAIT_ID]], hint={{[0-9]+}}, impl={{[0-9]+}}, return_address={{0x[0-f]+}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_acquired_nest_lock_first: wait_id=[[WAIT_ID]], return_address={{0x[0-f]+}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_wait_nest_lock: wait_id=[[WAIT_ID]], hint={{[0-9]+}}, impl={{[0-9]+}}, return_address={{0x[0-f]+}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_acquired_nest_lock_next: wait_id=[[WAIT_ID]], return_address={{0x[0-f]+}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_release_nest_lock_prev: wait_id=[[WAIT_ID]], return_address={{0x[0-f]+}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_release_nest_lock_last: wait_id=[[WAIT_ID]], return_address={{0x[0-f]+}}

  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_wait_critical: wait_id=[[WAIT_ID:[0-9]+]], hint={{[0-9]+}}, impl={{[0-9]+}}, return_address={{0x[0-f]+}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_acquired_critical: wait_id=[[WAIT_ID]], return_address={{0x[0-f]+}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_release_critical: wait_id=[[WAIT_ID]], return_address={{0x[0-f]+}} 

  // atomic cannot be tested because it is implemented with atomic hardware instructions
  // disabled_CHECK: {{^}}[[MASTER_ID]]: ompt_event_wait_atomic: wait_id=[[WAIT_ID:[0-9]+]], hint={{[0-9]+}}, impl={{[0-9]+}}, return_address={{0x[0-f]+}}
  // disabled_CHECK: {{^}}[[MASTER_ID]]: ompt_event_acquired_atomic: wait_id=[[WAIT_ID]], return_address={{0x[0-f]+}}

  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_wait_ordered: wait_id=[[WAIT_ID:[0-9]+]], hint={{[0-9]+}}, impl={{[0-9]+}}, return_address={{0x[0-f]+}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_acquired_ordered: wait_id=[[WAIT_ID]], return_address={{0x[0-f]+}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_release_ordered: wait_id=[[WAIT_ID]], return_address={{0x[0-f]+}} 

  return 0;
}
