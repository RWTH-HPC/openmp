// RUN: true
#include <stdio.h>
#include <inttypes.h>
#include <omp.h>
#include <ompt.h>
#include <execinfo.h>

static ompt_set_callback_t ompt_set_callback;
static ompt_get_thread_data_t ompt_get_thread_data;
static ompt_get_unique_id_t ompt_get_unique_id;


static int
on_ompt_callback_control_tool(
  uint64_t command,
  uint64_t modifier,
  void *arg,
  const void *codeptr_ra)
{
  printf("%" PRIu64 ": ompt_event_control_tool: command=%" PRIu64 ", modifier=%" PRIu64 ", arg=%p, codeptr_ra=%p\n", ompt_get_thread_data()->value, command, modifier, arg, codeptr_ra);
  return 0; //success
}

#define register_callback_t(name, type)                       \
do{                                                           \
  type f_##name = &on_##name;                                 \
  if (ompt_set_callback(name, (ompt_callback_t)f_##name) ==   \
      ompt_set_never)                                         \
    printf("0: Could not register callback '" #name "'\n");   \
}while(0)

#define register_callback(name) register_callback_t(name, name##_t)

int ompt_initialize(
  ompt_function_lookup_t lookup,
  ompt_fns_t* fns)
{
  ompt_set_callback = (ompt_set_callback_t) lookup("ompt_set_callback");
  ompt_get_thread_data = (ompt_get_thread_data_t) lookup("ompt_get_thread_data");
  ompt_get_unique_id = (ompt_get_unique_id_t) lookup("ompt_get_unique_id");


  register_callback(ompt_callback_control_tool);
  printf("0: NULL_POINTER=%p\n", (void*)NULL);
  return 1; //success
}

void ompt_finalize(ompt_fns_t* fns)
{
  printf("%d: ompt_event_runtime_shutdown\n", omp_get_thread_num());
}

ompt_fns_t* ompt_start_tool(
  unsigned int omp_version,
  const char *runtime_version)
{
  static ompt_fns_t ompt_fns = {&ompt_initialize,&ompt_finalize};
  return &ompt_fns;
}
