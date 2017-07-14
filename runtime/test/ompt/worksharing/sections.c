// RUN: %libomp-compile-and-run | %filecheck %s
// REQUIRES: ompt
#include "callback.h"
#include <omp.h>

int main()
{
  #pragma omp parallel sections num_threads(2)
  {
    #pragma omp section
    {
      printf("%lu: section 1\n", ompt_get_thread_data()->value);
    }
    #pragma omp section
    {
      printf("%lu: section 2\n", ompt_get_thread_data()->value);
    }
  }

  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_work'

  // CHECK: 0: NULL_POINTER=[[NULL:.*$]]
  //TODO: add test (for now: cannot distinguish between static for and sections)

  return 0;
}
