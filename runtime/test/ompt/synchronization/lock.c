// RUN: %libomp-compile-and-run | %filecheck %s
// REQUIRES: ompt
#include "callback.h"
#include <omp.h>

int main()
{
  //need to use an OpenMP construct so that OMPT will be initalized
  #pragma omp parallel num_threads(1)
    print_ids(0);

  omp_lock_t lock;
  printf("%" PRIu64 ": &lock: %lli\n", ompt_get_thread_data()->value, (long long) &lock);
  omp_init_lock(&lock);
  omp_set_lock(&lock);
  omp_unset_lock(&lock);
  omp_destroy_lock(&lock);


  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_mutex_acquire'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_mutex_acquired'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_mutex_released'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_nest_lock'

  // CHECK: 0: NULL_POINTER=[[NULL:.*$]]

  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: &lock: [[WAIT_ID:[0-9]+]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_init_lock: wait_id=[[WAIT_ID]], hint={{[0-9]+}}, impl={{[0-9]+}}, codeptr_ra={{0x[0-f]+}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_wait_lock: wait_id=[[WAIT_ID]], hint={{[0-9]+}}, impl={{[0-9]+}}, codeptr_ra={{0x[0-f]+}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_acquired_lock: wait_id=[[WAIT_ID]], codeptr_ra={{0x[0-f]+}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_release_lock: wait_id=[[WAIT_ID]], codeptr_ra={{0x[0-f]+}}

  return 0;
}
