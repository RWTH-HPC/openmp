#include <stdio.h>
#include <inttypes.h>
#include <ompt.h>

static ompt_get_task_data_t ompt_get_task_data;
static ompt_get_task_frame_t ompt_get_task_frame;
static ompt_get_thread_data_t ompt_get_thread_data;
static ompt_get_parallel_data_t ompt_get_parallel_data;

static int my_next_id()
{
//TODO: make sure this is thread-safe!
  static int ID=1;
  return __sync_fetch_and_add(&ID,1);
}

static void print_ids(int level)
{
  ompt_frame_t* frame = ompt_get_task_frame(level);
  if (frame)
    printf("%" PRIu64 ": level %d: parallel_id=%" PRIu64 ", task_id=%" PRIu64 ", exit_frame=%p, reenter_frame=%p\n", ompt_get_thread_data().value, level, ompt_get_parallel_data(level).value, ompt_get_task_data(level).value, frame->exit_runtime_frame, frame->reenter_runtime_frame);
  else
    printf("%" PRIu64 ": level %d: parallel_id=%" PRIu64 ", task_id=%" PRIu64 ", frame=%p\n", ompt_get_thread_data().value, level, ompt_get_parallel_data(level).value, ompt_get_task_data(level).value,               frame);
}

#define print_frame(level)\
do {\
  printf("%" PRIu64 ": __builtin_frame_address(%d)=%p\n", ompt_get_thread_data().value, level, __builtin_frame_address(level));\
} while(0)


static void
on_ompt_event_barrier_begin(
  ompt_parallel_data_t parallel_data,
  ompt_task_data_t task_data)
{
	printf("%" PRIu64 ": ompt_event_barrier_begin: parallel_id=%" PRIu64 ", task_id=%" PRIu64 "\n", ompt_get_thread_data().value, parallel_data.value, task_data.value);
  print_ids(0);
}

static void
on_ompt_event_barrier_end(
	ompt_parallel_data_t parallel_data,
  ompt_task_data_t task_data)
{
	printf("%" PRIu64 ": ompt_event_barrier_end: parallel_id=%" PRIu64 ", task_id=%" PRIu64 "\n", ompt_get_thread_data().value, parallel_data.value, task_data.value);
}

static void
on_ompt_event_implicit_task_begin(
	ompt_parallel_data_t parallel_data,
	ompt_task_data_t* task_data)
{
        task_data->value = my_next_id();
	printf("%" PRIu64 ": ompt_event_implicit_task_begin: parallel_id=%" PRIu64 ", task_id=%" PRIu64 "\n", ompt_get_thread_data().value, parallel_data.value, task_data->value);
}

static void
on_ompt_event_implicit_task_end(
	ompt_parallel_data_t parallel_data,
	ompt_task_data_t task_data)
{
	printf("%" PRIu64 ": ompt_event_implicit_task_end: parallel_id=%" PRIu64 ", task_id=%" PRIu64 "\n", ompt_get_thread_data().value, parallel_data.value, task_data.value);
}

static void
on_ompt_event_task_begin(
    ompt_task_data_t parent_task_data,    /* id of parent task            */
    ompt_frame_t *parent_task_frame,  /* frame data for parent task   */
    ompt_task_data_t*  new_task_data,      /* id of created task           */
    void *task_function)               /* pointer to outlined function */
{
        new_task_data->value = my_next_id();
  printf("%" PRIu64 ": ompt_event_task_create: parent_task_id=%" PRIu64 ", parent_task_frame.exit=%p, parent_task_frame.reenter=%p, new_task_id=%" PRIu64 ", parallel_function=%p\n", ompt_get_thread_data().value, parent_task_data.value, parent_task_frame->exit_runtime_frame, parent_task_frame->reenter_runtime_frame, new_task_data->value, task_function);
}

static void
on_ompt_event_task_switch(
    ompt_task_data_t first_task_data,
    ompt_task_data_t second_task_data)
{
  printf("%" PRIu64 ": ompt_event_task_schedule: first_task_id=%" PRIu64 ", second_task_id=%" PRIu64 "\n", ompt_get_thread_data().value, first_task_data.value, second_task_data.value);
}

static void
on_ompt_event_task_end(
    ompt_task_data_t task_data)            /* id of task                   */
{
  printf("%" PRIu64 ": ompt_event_task_end: task_id=%" PRIu64 "\n", ompt_get_thread_data().value, task_data.value);
}


static void
on_ompt_event_loop_begin(
  ompt_parallel_data_t parallel_data,
  ompt_task_data_t parent_task_data,
  void *workshare_function)
{
	printf("%" PRIu64 ": ompt_event_loop_begin: parallel_id=%" PRIu64 ", parent_task_id=%" PRIu64 ", workshare_function=%p\n", ompt_get_thread_data().value, parallel_data.value, parent_task_data.value, workshare_function);
}

static void
on_ompt_event_loop_end(
	ompt_parallel_data_t parallel_data,
	ompt_task_data_t task_data)
{
	printf("%" PRIu64 ": ompt_event_loop_end: parallel_id=%" PRIu64 ", task_id=%" PRIu64 "\n", ompt_get_thread_data().value, parallel_data.value, task_data.value);
}

static void
on_ompt_event_parallel_begin(
  ompt_task_data_t parent_task_data,
  ompt_frame_t *parent_task_frame,
  ompt_parallel_data_t* parallel_data,
  uint32_t requested_team_size,
  void *parallel_function,
  ompt_invoker_t invoker)
{
        parallel_data->value = my_next_id();
	printf("%" PRIu64 ": ompt_event_parallel_begin: parent_task_id=%" PRIu64 ", parent_task_frame.exit=%p, parent_task_frame.reenter=%p, parallel_id=%" PRIu64 ", requested_team_size=%" PRIu32 ", parallel_function=%p, invoker=%d\n", ompt_get_thread_data().value, parent_task_data.value, parent_task_frame->exit_runtime_frame, parent_task_frame->reenter_runtime_frame, parallel_data->value, requested_team_size, parallel_function, invoker);
}

static void
on_ompt_event_parallel_end(
	ompt_parallel_data_t parallel_data,
	ompt_task_data_t task_data,
  ompt_invoker_t invoker)
{
	printf("%" PRIu64 ": ompt_event_parallel_end: parallel_id=%" PRIu64 ", task_id=%" PRIu64 ", invoker=%d\n", ompt_get_thread_data().value, parallel_data.value, task_data.value, invoker);
}


void ompt_initialize(
  ompt_function_lookup_t lookup,
  const char *runtime_version,
  unsigned int ompt_version)
{
  ompt_set_callback_t ompt_set_callback = (ompt_set_callback_t) lookup("ompt_set_callback");
  ompt_get_task_data = (ompt_get_task_data_t) lookup("ompt_get_task_data");
  ompt_get_task_frame = (ompt_get_task_frame_t) lookup("ompt_get_task_frame");
  ompt_get_thread_data = (ompt_get_thread_data_t) lookup("ompt_get_thread_data");
  ompt_get_parallel_data = (ompt_get_parallel_data_t) lookup("ompt_get_parallel_data");

  ompt_set_callback(ompt_event_barrier_begin, (ompt_callback_t) &on_ompt_event_barrier_begin);
  ompt_set_callback(ompt_event_barrier_end, (ompt_callback_t) &on_ompt_event_barrier_end);
  ompt_set_callback(ompt_event_implicit_task_begin, (ompt_callback_t) &on_ompt_event_implicit_task_begin);
  ompt_set_callback(ompt_event_implicit_task_end, (ompt_callback_t) &on_ompt_event_implicit_task_end);
  ompt_set_callback(ompt_event_task_begin, (ompt_callback_t) &on_ompt_event_task_begin);
  ompt_set_callback(ompt_event_task_switch, (ompt_callback_t) &on_ompt_event_task_switch);
  ompt_set_callback(ompt_event_task_end, (ompt_callback_t) &on_ompt_event_task_end);
  ompt_set_callback(ompt_event_loop_begin, (ompt_callback_t) &on_ompt_event_loop_begin);
  ompt_set_callback(ompt_event_loop_end, (ompt_callback_t) &on_ompt_event_loop_end);
  ompt_set_callback(ompt_event_parallel_begin, (ompt_callback_t) &on_ompt_event_parallel_begin);
  ompt_set_callback(ompt_event_parallel_end, (ompt_callback_t) &on_ompt_event_parallel_end);
  printf("0: NULL_POINTER=%p\n", NULL);
}

ompt_initialize_t ompt_tool()
{
  return &ompt_initialize;
}
