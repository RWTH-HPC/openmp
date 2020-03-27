/*
 * kmp_tasking.cpp -- OpenMP 3.0 tasking support.
 */


//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//


#include "kmp.h"
#include "kmp_i18n.h"
#include "kmp_itt.h"
#include "kmp_stats.h"
#include "kmp_wait_release.h"

#if OMPT_SUPPORT
#include "ompt-specific.h"
#endif

#include "tsan_annotations.h"

#if KMP_USE_TASK_AFFINITY
#include <numa.h>
#include <numaif.h>
#include <unistd.h>
#include <limits.h>
#endif

/* forward declaration */
static void __kmp_enable_tasking(kmp_task_team_t *task_team,
                                 kmp_info_t *this_thr);
static void __kmp_alloc_task_deque(kmp_info_t *thread,
                                   kmp_thread_data_t *thread_data);
static int __kmp_realloc_task_threads_data(kmp_info_t *thread,
                                           kmp_task_team_t *task_team);
static bool __kmp_give_task(kmp_info_t *thread, kmp_int32 tid, kmp_task_t *task,
                            kmp_int32 pass);
static void __kmp_realloc_task_deque(kmp_info_t *thread,
                                     kmp_thread_data_t *thread_data);

#if KMP_USE_TASK_AFFINITY
inline bool __kmp_task_aff_is_correct_task(
  kmp_info_t * thread,
  kmp_taskdata_t *taskdata,
  kmp_thread_data_t *victim_td,
  kmp_task_team_t * task_team);

inline kmp_info_t * __kmp_task_aff_get_initial_thread_in_numa_domain(
  int current_data_domain,
  kmp_task_team_t *task_team,
  kmp_thread_data_t *threads_data,
  int* target_tid,
  int* target_gtid);

kmp_int32 __kmp_omp_task_aff(kmp_int32 gtid, kmp_int32 target_gtid, kmp_task_t *new_task,
                         bool serialize_immediate);
kmp_int32 __kmpc_omp_task_affinity(ident_t *loc_ref, kmp_int32 gtid, kmp_task_t *new_task);
#endif

#ifdef OMP_45_ENABLED
static void __kmp_bottom_half_finish_proxy(kmp_int32 gtid, kmp_task_t *ptask);
#endif

#ifdef BUILD_TIED_TASK_STACK

//  __kmp_trace_task_stack: print the tied tasks from the task stack in order
//  from top do bottom
//
//  gtid: global thread identifier for thread containing stack
//  thread_data: thread data for task team thread containing stack
//  threshold: value above which the trace statement triggers
//  location: string identifying call site of this function (for trace)
static void __kmp_trace_task_stack(kmp_int32 gtid,
                                   kmp_thread_data_t *thread_data,
                                   int threshold, char *location) {
  kmp_task_stack_t *task_stack = &thread_data->td.td_susp_tied_tasks;
  kmp_taskdata_t **stack_top = task_stack->ts_top;
  kmp_int32 entries = task_stack->ts_entries;
  kmp_taskdata_t *tied_task;

  KA_TRACE(
      threshold,
      ("__kmp_trace_task_stack(start): location = %s, gtid = %d, entries = %d, "
       "first_block = %p, stack_top = %p \n",
       location, gtid, entries, task_stack->ts_first_block, stack_top));

  KMP_DEBUG_ASSERT(stack_top != NULL);
  KMP_DEBUG_ASSERT(entries > 0);

  while (entries != 0) {
    KMP_DEBUG_ASSERT(stack_top != &task_stack->ts_first_block.sb_block[0]);
    // fix up ts_top if we need to pop from previous block
    if (entries & TASK_STACK_INDEX_MASK == 0) {
      kmp_stack_block_t *stack_block = (kmp_stack_block_t *)(stack_top);

      stack_block = stack_block->sb_prev;
      stack_top = &stack_block->sb_block[TASK_STACK_BLOCK_SIZE];
    }

    // finish bookkeeping
    stack_top--;
    entries--;

    tied_task = *stack_top;

    KMP_DEBUG_ASSERT(tied_task != NULL);
    KMP_DEBUG_ASSERT(tied_task->td_flags.tasktype == TASK_TIED);

    KA_TRACE(threshold,
             ("__kmp_trace_task_stack(%s):             gtid=%d, entry=%d, "
              "stack_top=%p, tied_task=%p\n",
              location, gtid, entries, stack_top, tied_task));
  }
  KMP_DEBUG_ASSERT(stack_top == &task_stack->ts_first_block.sb_block[0]);

  KA_TRACE(threshold,
           ("__kmp_trace_task_stack(exit): location = %s, gtid = %d\n",
            location, gtid));
}

//  __kmp_init_task_stack: initialize the task stack for the first time
//  after a thread_data structure is created.
//  It should not be necessary to do this again (assuming the stack works).
//
//  gtid: global thread identifier of calling thread
//  thread_data: thread data for task team thread containing stack
static void __kmp_init_task_stack(kmp_int32 gtid,
                                  kmp_thread_data_t *thread_data) {
  kmp_task_stack_t *task_stack = &thread_data->td.td_susp_tied_tasks;
  kmp_stack_block_t *first_block;

  // set up the first block of the stack
  first_block = &task_stack->ts_first_block;
  task_stack->ts_top = (kmp_taskdata_t **)first_block;
  memset((void *)first_block, '\0',
         TASK_STACK_BLOCK_SIZE * sizeof(kmp_taskdata_t *));

  // initialize the stack to be empty
  task_stack->ts_entries = TASK_STACK_EMPTY;
  first_block->sb_next = NULL;
  first_block->sb_prev = NULL;
}

//  __kmp_free_task_stack: free the task stack when thread_data is destroyed.
//
//  gtid: global thread identifier for calling thread
//  thread_data: thread info for thread containing stack
static void __kmp_free_task_stack(kmp_int32 gtid,
                                  kmp_thread_data_t *thread_data) {
  kmp_task_stack_t *task_stack = &thread_data->td.td_susp_tied_tasks;
  kmp_stack_block_t *stack_block = &task_stack->ts_first_block;

  KMP_DEBUG_ASSERT(task_stack->ts_entries == TASK_STACK_EMPTY);
  // free from the second block of the stack
  while (stack_block != NULL) {
    kmp_stack_block_t *next_block = (stack_block) ? stack_block->sb_next : NULL;

    stack_block->sb_next = NULL;
    stack_block->sb_prev = NULL;
    if (stack_block != &task_stack->ts_first_block) {
      __kmp_thread_free(thread,
                        stack_block); // free the block, if not the first
    }
    stack_block = next_block;
  }
  // initialize the stack to be empty
  task_stack->ts_entries = 0;
  task_stack->ts_top = NULL;
}

//  __kmp_push_task_stack: Push the tied task onto the task stack.
//     Grow the stack if necessary by allocating another block.
//
//  gtid: global thread identifier for calling thread
//  thread: thread info for thread containing stack
//  tied_task: the task to push on the stack
static void __kmp_push_task_stack(kmp_int32 gtid, kmp_info_t *thread,
                                  kmp_taskdata_t *tied_task) {
  // GEH - need to consider what to do if tt_threads_data not allocated yet
  kmp_thread_data_t *thread_data =
      &thread->th.th_task_team->tt.tt_threads_data[__kmp_tid_from_gtid(gtid)];
  kmp_task_stack_t *task_stack = &thread_data->td.td_susp_tied_tasks;

  if (tied_task->td_flags.team_serial || tied_task->td_flags.tasking_ser) {
    return; // Don't push anything on stack if team or team tasks are serialized
  }

  KMP_DEBUG_ASSERT(tied_task->td_flags.tasktype == TASK_TIED);
  KMP_DEBUG_ASSERT(task_stack->ts_top != NULL);

  KA_TRACE(20,
           ("__kmp_push_task_stack(enter): GTID: %d; THREAD: %p; TASK: %p\n",
            gtid, thread, tied_task));
  // Store entry
  *(task_stack->ts_top) = tied_task;

  // Do bookkeeping for next push
  task_stack->ts_top++;
  task_stack->ts_entries++;

  if (task_stack->ts_entries & TASK_STACK_INDEX_MASK == 0) {
    // Find beginning of this task block
    kmp_stack_block_t *stack_block =
        (kmp_stack_block_t *)(task_stack->ts_top - TASK_STACK_BLOCK_SIZE);

    // Check if we already have a block
    if (stack_block->sb_next !=
        NULL) { // reset ts_top to beginning of next block
      task_stack->ts_top = &stack_block->sb_next->sb_block[0];
    } else { // Alloc new block and link it up
      kmp_stack_block_t *new_block = (kmp_stack_block_t *)__kmp_thread_calloc(
          thread, sizeof(kmp_stack_block_t));

      task_stack->ts_top = &new_block->sb_block[0];
      stack_block->sb_next = new_block;
      new_block->sb_prev = stack_block;
      new_block->sb_next = NULL;

      KA_TRACE(
          30,
          ("__kmp_push_task_stack(): GTID: %d; TASK: %p; Alloc new block: %p\n",
           gtid, tied_task, new_block));
    }
  }
  KA_TRACE(20, ("__kmp_push_task_stack(exit): GTID: %d; TASK: %p\n", gtid,
                tied_task));
}

//  __kmp_pop_task_stack: Pop the tied task from the task stack.  Don't return
//  the task, just check to make sure it matches the ending task passed in.
//
//  gtid: global thread identifier for the calling thread
//  thread: thread info structure containing stack
//  tied_task: the task popped off the stack
//  ending_task: the task that is ending (should match popped task)
static void __kmp_pop_task_stack(kmp_int32 gtid, kmp_info_t *thread,
                                 kmp_taskdata_t *ending_task) {
  // GEH - need to consider what to do if tt_threads_data not allocated yet
  kmp_thread_data_t *thread_data =
      &thread->th.th_task_team->tt_threads_data[__kmp_tid_from_gtid(gtid)];
  kmp_task_stack_t *task_stack = &thread_data->td.td_susp_tied_tasks;
  kmp_taskdata_t *tied_task;

  if (ending_task->td_flags.team_serial || ending_task->td_flags.tasking_ser) {
    // Don't pop anything from stack if team or team tasks are serialized
    return;
  }

  KMP_DEBUG_ASSERT(task_stack->ts_top != NULL);
  KMP_DEBUG_ASSERT(task_stack->ts_entries > 0);

  KA_TRACE(20, ("__kmp_pop_task_stack(enter): GTID: %d; THREAD: %p\n", gtid,
                thread));

  // fix up ts_top if we need to pop from previous block
  if (task_stack->ts_entries & TASK_STACK_INDEX_MASK == 0) {
    kmp_stack_block_t *stack_block = (kmp_stack_block_t *)(task_stack->ts_top);

    stack_block = stack_block->sb_prev;
    task_stack->ts_top = &stack_block->sb_block[TASK_STACK_BLOCK_SIZE];
  }

  // finish bookkeeping
  task_stack->ts_top--;
  task_stack->ts_entries--;

  tied_task = *(task_stack->ts_top);

  KMP_DEBUG_ASSERT(tied_task != NULL);
  KMP_DEBUG_ASSERT(tied_task->td_flags.tasktype == TASK_TIED);
  KMP_DEBUG_ASSERT(tied_task == ending_task); // If we built the stack correctly

  KA_TRACE(20, ("__kmp_pop_task_stack(exit): GTID: %d; TASK: %p\n", gtid,
                tied_task));
  return;
}
#endif /* BUILD_TIED_TASK_STACK */

#if KMP_USE_TASK_AFFINITY
kmp_maphash_t *__kmp_maphash_create(kmp_info_t *thread) {
  kmp_maphash_t *h;

  size_t h_size;
  h_size = KMP_MAPHASH_MASTER_SIZE;
  //h_size = KMP_DEPHASH_OTHER_SIZE;

  kmp_int32 size = h_size * sizeof(kmp_maphash_entry_t *) + sizeof(kmp_maphash_t);

#if USE_FAST_MEMORY
  h = (kmp_maphash_t *)__kmp_fast_allocate(thread, size);
#else
  h = (kmp_maphash_t *)__kmp_thread_malloc(thread, size);
#endif
  h->size = h_size;

#ifdef KMP_DEBUG
  h->nelements = 0;
  h->nconflicts = 0;
#endif
  h->buckets = (kmp_maphash_entry **)(h + 1);

  for (size_t i = 0; i < h_size; i++)
    h->buckets[i] = 0;

  return h;
}

inline kmp_int32 __kmp_maphash_hash(kmp_intptr_t addr, size_t hsize) {
  // TODO alternate to try: set = (((Addr64)(addrUsefulBits * 9.618)) %
  // m_num_sets );
  return ((addr >> 6) ^ (addr >> 2)) % hsize;
}

kmp_maphash_entry * __kmp_maphash_find(kmp_info_t *thread, kmp_maphash_t *h, kmp_intptr_t addr) {
  kmp_int32 bucket = __kmp_maphash_hash(addr, h->size);

  kmp_maphash_entry_t *entry;
  for (entry = h->buckets[bucket]; entry; entry = entry->next_in_bucket)
    if (entry->addr == addr)
      break;

  if (entry == NULL) {
// create entry. This is only done by one thread so no locking required
// MAYBE THIS MIGHT CHANGE: bucket lock
#if USE_FAST_MEMORY
    entry = (kmp_maphash_entry_t *)__kmp_fast_allocate(thread, sizeof(kmp_maphash_entry_t));
#else
    entry = (kmp_maphash_entry_t *)__kmp_thread_malloc(thread, sizeof(kmp_maphash_entry_t));
#endif
    entry->addr = addr;
    entry->val.data_domain = -1;
    entry->val.gtid = -1;
    // entry->last_out = NULL;
    // entry->last_ins = NULL;
    entry->next_in_bucket = h->buckets[bucket];
    h->buckets[bucket] = entry; //WTF? ref to self?
#ifdef KMP_DEBUG
    h->nelements++;
    if (entry->next_in_bucket)
      h->nconflicts++;
#endif
  }
  return entry;
}

//  __kmp_push_task: Add a task to the thread's deque
static kmp_int32 __kmp_push_task_aff(kmp_int32 gtid, kmp_int32 orig_id, kmp_task_t *task) {
  kmp_info_t *thread = __kmp_threads[gtid];
  kmp_info_t *orig_thread = __kmp_threads[orig_id];
  kmp_taskdata_t *taskdata = KMP_TASK_TO_TASKDATA(task);
  kmp_task_team_t *task_team = orig_thread->th.th_task_team;
  kmp_int32 tid = __kmp_tid_from_gtid(gtid);
  kmp_thread_data_t *thread_data;

  KA_TRACE(20,
           ("__kmp_push_task_aff: T#%d trying to push task %p to thread T#%d.\n", orig_id, taskdata, gtid));

  if (taskdata->td_flags.tiedness == TASK_UNTIED) {
    // untied task needs to increment counter so that the task structure is not
    // freed prematurely
    kmp_int32 counter = 1 + KMP_TEST_THEN_INC32(&taskdata->td_untied_count);
    KA_TRACE(
        20,
        ("__kmp_push_task_aff: T#%d untied_count (%d) incremented for task %p\n",
         orig_id, counter, taskdata));
  }

  // The first check avoids building task_team thread data if serialized
  if (taskdata->td_flags.task_serial) {
    KA_TRACE(20, ("__kmp_push_task_aff: T#%d team serialized; returning "
                  "TASK_NOT_PUSHED for task %p\n",
                  orig_id, taskdata));
    return TASK_NOT_PUSHED;
  }

  // Now that serialized tasks have returned, we can assume that we are not in
  // immediate exec mode
  if(task_team == NULL)
    return TASK_NOT_PUSHED;

  // KMP_DEBUG_ASSERT(__kmp_tasking_mode != tskm_immediate_exec);
  // if (!KMP_TASKING_ENABLED(task_team)) {
  //   __kmp_enable_tasking(task_team, thread);
  // }
  KMP_DEBUG_ASSERT(TCR_4(task_team->tt.tt_found_tasks) == TRUE);
  KMP_DEBUG_ASSERT(TCR_PTR(task_team->tt.tt_threads_data) != NULL);

  // Find tasking deque specific to encountering thread
  thread_data = &task_team->tt.tt_threads_data[tid];

  //volatile kmp_taskdata_t *** tmp_deque_check = &thread_data->td.td_deque;
  //while(*tmp_deque_check == NULL){
  // while(thread_data->td.td_deque == NULL){
  //   // wait as long as target thread has initialized its queue
  // }
  if (thread_data->td.td_deque == NULL) {
    return TASK_NOT_PUSHED;
    // __kmp_acquire_bootstrap_lock(&thread_data->td.td_deque_lock);
    // __kmp_alloc_task_deque(thread, thread_data);
    // __kmp_release_bootstrap_lock(&thread_data->td.td_deque_lock);
  }

  // Check if deque is full
  if (TCR_4(thread_data->td.td_deque_ntasks) >=
      TASK_DEQUE_SIZE(thread_data->td)) {
    KA_TRACE(20, ("__kmp_push_task_aff: T#%d deque of T#%d is full; returning "
                  "TASK_NOT_PUSHED for task %p, deque size=%d\n",
                  orig_id, gtid, taskdata, TASK_DEQUE_SIZE(thread_data->td)));

    // increase size
    // __kmp_acquire_bootstrap_lock(&thread_data->td.td_deque_lock);
    // __kmp_realloc_task_deque(thread, thread_data);
    // __kmp_release_bootstrap_lock(&thread_data->td.td_deque_lock);
    return TASK_NOT_PUSHED;
  }

  // Lock the deque for the task push operation
  __kmp_acquire_bootstrap_lock(&thread_data->td.td_deque_lock);

#if OMP_45_ENABLED
  // Need to recheck as we can get a proxy task from a thread outside of OpenMP
  if (TCR_4(thread_data->td.td_deque_ntasks) >=
      TASK_DEQUE_SIZE(thread_data->td)) {
    //__kmp_realloc_task_deque(thread, thread_data);
    __kmp_release_bootstrap_lock(&thread_data->td.td_deque_lock);
    KA_TRACE(20, ("__kmp_push_task_aff: T#%d deque of T#%d is full on 2nd check; returning "
                  "TASK_NOT_PUSHED for task %p, deque size=%d\n",
                  orig_id, gtid, taskdata, TASK_DEQUE_SIZE(thread_data->td)));
    return TASK_NOT_PUSHED;
  }
#else
  // Must have room since no thread can add tasks but calling thread
  KMP_DEBUG_ASSERT(TCR_4(thread_data->td.td_deque_ntasks) <
                   TASK_DEQUE_SIZE(thread_data->td));
#endif

  thread_data->td.td_deque[thread_data->td.td_deque_tail] =
      taskdata; // Push taskdata
  // Wrap index.
  thread_data->td.td_deque_tail =
      (thread_data->td.td_deque_tail + 1) & TASK_DEQUE_MASK(thread_data->td);
  TCW_4(thread_data->td.td_deque_ntasks,
        TCR_4(thread_data->td.td_deque_ntasks) + 1); // Adjust task count

  KA_TRACE(20, ("__kmp_push_task_aff: T#%d returning TASK_SUCCESSFULLY_PUSHED to T#%d: "
                "task=%p ntasks=%d head=%u tail=%u\n",
                orig_id, gtid, taskdata, thread_data->td.td_deque_ntasks,
                thread_data->td.td_deque_head, thread_data->td.td_deque_tail));

  taskdata->td_task_affinity_scheduled_thread = gtid;
  taskdata->td_task_affinity_scheduled_thread_set = true;
  orig_thread->th.th_count_task_with_affinity_generated++;
  __kmp_release_bootstrap_lock(&thread_data->td.td_deque_lock);

  return TASK_SUCCESSFULLY_PUSHED;
}
#endif

//  __kmp_push_task: Add a task to the thread's deque
static kmp_int32 __kmp_push_task(kmp_int32 gtid, kmp_task_t *task) {
  kmp_info_t *thread = __kmp_threads[gtid];
  kmp_taskdata_t *taskdata = KMP_TASK_TO_TASKDATA(task);
  kmp_task_team_t *task_team = thread->th.th_task_team;
  kmp_int32 tid = __kmp_tid_from_gtid(gtid);
  kmp_thread_data_t *thread_data;

  KA_TRACE(20,
           ("__kmp_push_task: T#%d trying to push task %p.\n", gtid, taskdata));

  if (taskdata->td_flags.tiedness == TASK_UNTIED) {
    // untied task needs to increment counter so that the task structure is not
    // freed prematurely
    kmp_int32 counter = 1 + KMP_TEST_THEN_INC32(&taskdata->td_untied_count);
    KA_TRACE(
        20,
        ("__kmp_push_task: T#%d untied_count (%d) incremented for task %p\n",
         gtid, counter, taskdata));
  }

  // The first check avoids building task_team thread data if serialized
  if (taskdata->td_flags.task_serial) {
    KA_TRACE(20, ("__kmp_push_task: T#%d team serialized; returning "
                  "TASK_NOT_PUSHED for task %p\n",
                  gtid, taskdata));
    return TASK_NOT_PUSHED;
  }

  // Now that serialized tasks have returned, we can assume that we are not in
  // immediate exec mode
  KMP_DEBUG_ASSERT(__kmp_tasking_mode != tskm_immediate_exec);
  if (!KMP_TASKING_ENABLED(task_team)) {
    __kmp_enable_tasking(task_team, thread);
  }
  KMP_DEBUG_ASSERT(TCR_4(task_team->tt.tt_found_tasks) == TRUE);
  KMP_DEBUG_ASSERT(TCR_PTR(task_team->tt.tt_threads_data) != NULL);

  // Find tasking deque specific to encountering thread
  thread_data = &task_team->tt.tt_threads_data[tid];

  // No lock needed since only owner can allocate
  if (thread_data->td.td_deque == NULL) {
    __kmp_alloc_task_deque(thread, thread_data);
  }

  // Check if deque is full
  if (TCR_4(thread_data->td.td_deque_ntasks) >=
      TASK_DEQUE_SIZE(thread_data->td)) {
    KA_TRACE(20, ("__kmp_push_task: T#%d deque is full; returning "
                  "TASK_NOT_PUSHED for task %p\n",
                  gtid, taskdata));
    return TASK_NOT_PUSHED;
  }

  // Lock the deque for the task push operation
  __kmp_acquire_bootstrap_lock(&thread_data->td.td_deque_lock);

#if OMP_45_ENABLED
  // Need to recheck as we can get a proxy task from a thread outside of OpenMP
  if (TCR_4(thread_data->td.td_deque_ntasks) >=
      TASK_DEQUE_SIZE(thread_data->td)) {
    __kmp_release_bootstrap_lock(&thread_data->td.td_deque_lock);
    KA_TRACE(20, ("__kmp_push_task: T#%d deque is full on 2nd check; returning "
                  "TASK_NOT_PUSHED for task %p\n",
                  gtid, taskdata));
    return TASK_NOT_PUSHED;
  }
#else
  // Must have room since no thread can add tasks but calling thread
  KMP_DEBUG_ASSERT(TCR_4(thread_data->td.td_deque_ntasks) <
                   TASK_DEQUE_SIZE(thread_data->td));
#endif

  thread_data->td.td_deque[thread_data->td.td_deque_tail] =
      taskdata; // Push taskdata
  // Wrap index.
  thread_data->td.td_deque_tail =
      (thread_data->td.td_deque_tail + 1) & TASK_DEQUE_MASK(thread_data->td);
  TCW_4(thread_data->td.td_deque_ntasks,
        TCR_4(thread_data->td.td_deque_ntasks) + 1); // Adjust task count

  KA_TRACE(20, ("__kmp_push_task: T#%d returning TASK_SUCCESSFULLY_PUSHED: "
                "task=%p ntasks=%d head=%u tail=%u\n",
                gtid, taskdata, thread_data->td.td_deque_ntasks,
                thread_data->td.td_deque_head, thread_data->td.td_deque_tail));

#if KMP_USE_TASK_AFFINITY
  if(taskdata->naffin != 0)
  {
    thread->th.th_count_task_with_affinity_generated++;
  }
  taskdata->td_task_affinity_scheduled_thread = gtid;
  taskdata->td_task_affinity_scheduled_thread_set = true;
#endif
  __kmp_release_bootstrap_lock(&thread_data->td.td_deque_lock);
  return TASK_SUCCESSFULLY_PUSHED;
}

// __kmp_pop_current_task_from_thread: set up current task from called thread
// when team ends
//
// this_thr: thread structure to set current_task in.
void __kmp_pop_current_task_from_thread(kmp_info_t *this_thr) {
  KF_TRACE(10, ("__kmp_pop_current_task_from_thread(enter): T#%d "
                "this_thread=%p, curtask=%p, "
                "curtask_parent=%p\n",
                0, this_thr, this_thr->th.th_current_task,
                this_thr->th.th_current_task->td_parent));

  this_thr->th.th_current_task = this_thr->th.th_current_task->td_parent;

  KF_TRACE(10, ("__kmp_pop_current_task_from_thread(exit): T#%d "
                "this_thread=%p, curtask=%p, "
                "curtask_parent=%p\n",
                0, this_thr, this_thr->th.th_current_task,
                this_thr->th.th_current_task->td_parent));
}

// __kmp_push_current_task_to_thread: set up current task in called thread for a
// new team
//
// this_thr: thread structure to set up
// team: team for implicit task data
// tid: thread within team to set up
void __kmp_push_current_task_to_thread(kmp_info_t *this_thr, kmp_team_t *team,
                                       int tid) {
  // current task of the thread is a parent of the new just created implicit
  // tasks of new team
  KF_TRACE(10, ("__kmp_push_current_task_to_thread(enter): T#%d this_thread=%p "
                "curtask=%p "
                "parent_task=%p\n",
                tid, this_thr, this_thr->th.th_current_task,
                team->t.t_implicit_task_taskdata[tid].td_parent));

  KMP_DEBUG_ASSERT(this_thr != NULL);

  if (tid == 0) {
    if (this_thr->th.th_current_task != &team->t.t_implicit_task_taskdata[0]) {
      team->t.t_implicit_task_taskdata[0].td_parent =
          this_thr->th.th_current_task;
      this_thr->th.th_current_task = &team->t.t_implicit_task_taskdata[0];
    }
  } else {
    team->t.t_implicit_task_taskdata[tid].td_parent =
        team->t.t_implicit_task_taskdata[0].td_parent;
    this_thr->th.th_current_task = &team->t.t_implicit_task_taskdata[tid];
  }

  KF_TRACE(10, ("__kmp_push_current_task_to_thread(exit): T#%d this_thread=%p "
                "curtask=%p "
                "parent_task=%p\n",
                tid, this_thr, this_thr->th.th_current_task,
                team->t.t_implicit_task_taskdata[tid].td_parent));
}

// __kmp_task_start: bookkeeping for a task starting execution
//
// GTID: global thread id of calling thread
// task: task starting execution
// current_task: task suspending
static void __kmp_task_start(kmp_int32 gtid, kmp_task_t *task,
                             kmp_taskdata_t *current_task) {
  kmp_taskdata_t *taskdata = KMP_TASK_TO_TASKDATA(task);
  kmp_info_t *thread = __kmp_threads[gtid];

  KA_TRACE(10,
           ("__kmp_task_start(enter_aff):\tT#%d\tstarting task %p: current_task=%p\n",
            gtid, taskdata, current_task));

  KMP_DEBUG_ASSERT(taskdata->td_flags.tasktype == TASK_EXPLICIT);

#if KMP_USE_TASK_AFFINITY
  bool startet_at_correct_thread = false;
  if(taskdata->td_task_affinity_scheduled_thread_set)
  {
    thread->th.th_count_task_with_affinity_started++;
    if(taskdata->td_task_affinity_scheduled_thread == gtid)
    {
      thread->th.th_count_task_started_at_correct_thread++;
      startet_at_correct_thread = true;
    }
    

    int tmp_domain = map_thread_to_numa_domain[taskdata->td_task_affinity_scheduled_thread];
    // get domain for thread
    if(thread->th.th_task_aff_my_domain_nr == tmp_domain)
    {
      thread->th.th_count_task_started_at_correct_threads_domain++;
    }
    if(taskdata->td_task_affinity_data_domain == thread->th.th_task_aff_my_domain_nr)
    {
      thread->th.th_count_task_started_at_correct_data_domain++;
    }
    else {
      if (taskdata->td_task_affinity_data_domain < 0) {
        KA_TRACE(3,("task_start: Invalid Data Domain"));
      } else { 
        KA_TRACE(3,("task_start: false Data Domain"));
      }
    }
  }
  if(taskdata->naffin != 0) {


  KA_TRACE(3, ("_kmp_task_start: Task starded in Thread: %d\n", gtid));
#if KMP_TASK_AFFINITY_MEASURE_TIME
      double time1;
      time1 = get_wall_time2();
#endif
  if(!startet_at_correct_thread)
  {
  #if KMP_TASK_AFFINITY_USE_DEFAULT_MAP
    __kmp_acquire_bootstrap_lock(&lock_addr_map);
    int j = 0;
    for (j = 0; j < taskdata->naffin; j++)
    {
      task_aff_addr_map[taskdata->affinity_info[j].base_addr].gtid = gtid;
    }
    __kmp_release_bootstrap_lock(&lock_addr_map);
  #else
    cur_entry->val.gtid = gtid;
  #endif
  }
  
#if KMP_TASK_AFFINITY_MEASURE_TIME
      time1 = get_wall_time2()-time1;
      thread->th.th_sum_time_map_insert += time1;
      thread->th.th_num_map_insert++;
#endif
  }
#endif

  // mark currently executing task as suspended
  // TODO: GEH - make sure root team implicit task is initialized properly.
  // KMP_DEBUG_ASSERT( current_task -> td_flags.executing == 1 );
  current_task->td_flags.executing = 0;

// Add task to stack if tied
#ifdef BUILD_TIED_TASK_STACK
  if (taskdata->td_flags.tiedness == TASK_TIED) {
    __kmp_push_task_stack(gtid, thread, taskdata);
  }
#endif /* BUILD_TIED_TASK_STACK */

  // mark starting task as executing and as current task
  thread->th.th_current_task = taskdata;

  KMP_DEBUG_ASSERT(taskdata->td_flags.started == 0 ||
                   taskdata->td_flags.tiedness == TASK_UNTIED);
  KMP_DEBUG_ASSERT(taskdata->td_flags.executing == 0 ||
                   taskdata->td_flags.tiedness == TASK_UNTIED);
  taskdata->td_flags.started = 1;
  taskdata->td_flags.executing = 1;
  KMP_DEBUG_ASSERT(taskdata->td_flags.complete == 0);
  KMP_DEBUG_ASSERT(taskdata->td_flags.freed == 0);

  // GEH TODO: shouldn't we pass some sort of location identifier here?
  // APT: yes, we will pass location here.
  // need to store current thread state (in a thread or taskdata structure)
  // before setting work_state, otherwise wrong state is set after end of task

  KA_TRACE(10, ("__kmp_task_start(exit): T#%d task=%p\n", gtid, taskdata));

  return;
}

#if OMPT_SUPPORT
//------------------------------------------------------------------------------
// __ompt_task_init:
//   Initialize OMPT fields maintained by a task. This will only be called after
//   ompt_tool, so we already know whether ompt is enabled or not.

static inline void __ompt_task_init(kmp_taskdata_t *task, int tid) {
// The calls to __ompt_task_init already have the ompt_enabled condition.
//  if (__builtin_expect(ompt_enabled.enabled,0)) {
  task->ompt_task_info.task_data.value = 0;
  task->ompt_task_info.frame.exit_runtime_frame = NULL;
  task->ompt_task_info.frame.reenter_runtime_frame = NULL;
#if OMP_40_ENABLED
  task->ompt_task_info.ndeps = 0;
  task->ompt_task_info.deps = NULL;
#endif /* OMP_40_ENABLED */
//  }
}

// __ompt_task_start:
//   Build and trigger task-begin event
static inline void __ompt_task_start(kmp_task_t *task, kmp_taskdata_t *current_task,
                       kmp_int32 gtid) {
// The calls to __ompt_task_start already have the ompt_enabled condition.
//  if (__builtin_expect(ompt_enabled.enabled,0)){
  kmp_taskdata_t *taskdata = KMP_TASK_TO_TASKDATA(task);
  ompt_task_status_t status = ompt_task_others;
  if (__kmp_threads[gtid]->th.ompt_thread_info.ompt_task_yielded) {
    status = ompt_task_yield;
    __kmp_threads[gtid]->th.ompt_thread_info.ompt_task_yielded = 0;
  }
  /* let OMPT know that we're about to run this task */
  if (ompt_enabled.ompt_callback_task_schedule) {
    ompt_callbacks.ompt_callback(ompt_callback_task_schedule)(
        &(current_task->ompt_task_info.task_data), status,
        &(taskdata->ompt_task_info.task_data));
  }
  taskdata->ompt_task_info.scheduling_parent = current_task;
//  }
}

// __ompt_task_start:
//   Build and trigger task-end event
static inline void __ompt_task_finish(kmp_task_t *task, kmp_taskdata_t *resumed_task) {
// The calls to __ompt_task_finish already have the ompt_enabled condition.
//  if (__builtin_expect(ompt_enabled.enabled,0)){
  kmp_taskdata_t *taskdata = KMP_TASK_TO_TASKDATA(task);
  ompt_task_status_t status = ompt_task_complete;
  if (taskdata->td_flags.tiedness == TASK_UNTIED &&
      KMP_TEST_THEN_ADD32(&(taskdata->td_untied_count), 0) > 1)
    status = ompt_task_others;
  if (__kmp_omp_cancellation && taskdata->td_taskgroup &&
      taskdata->td_taskgroup->cancel_request == cancel_taskgroup) {
    status = ompt_task_cancel;
  }

  /* let OMPT know that we're returning to the callee task */
  if (ompt_enabled.ompt_callback_task_schedule) {
    ompt_callbacks.ompt_callback(ompt_callback_task_schedule)(
        &(taskdata->ompt_task_info.task_data), status,
        &((resumed_task ? resumed_task
                        : (taskdata->ompt_task_info.scheduling_parent
                               ? taskdata->ompt_task_info.scheduling_parent
                               : taskdata->td_parent))
              ->ompt_task_info.task_data));
  }
//  }
}
#endif

#if OMPT_SUPPORT
OMPT_NOINLINE
static void __ompt_enabled_task_begin_if0(ident_t *loc_ref, kmp_int32 gtid,
    kmp_task_t *task, void *frame_address, void *return_address) {
  kmp_taskdata_t *taskdata = KMP_TASK_TO_TASKDATA(task);
  kmp_taskdata_t *current_task = __kmp_threads[gtid]->th.th_current_task;

  KA_TRACE(10, ("__kmpc_omp_task_begin_if0(enter): T#%d loc=%p task=%p "
                "current_task=%p\n",
                gtid, loc_ref, taskdata, current_task));

  if (taskdata->td_flags.tiedness == TASK_UNTIED) {
    // untied task needs to increment counter so that the task structure is not
    // freed prematurely
    kmp_int32 counter = 1 + KMP_TEST_THEN_INC32(&taskdata->td_untied_count);
    KA_TRACE(20, ("__kmpc_omp_task_begin_if0: T#%d untied_count (%d) "
                  "incremented for task %p\n",
                  gtid, counter, taskdata));
  }

  taskdata->td_flags.task_serial =
      1; // Execute this task immediately, not deferred.

  __kmp_task_start(gtid, task, current_task);

  if (current_task->ompt_task_info.frame.reenter_runtime_frame == NULL) {
    current_task->ompt_task_info.frame.reenter_runtime_frame =
        taskdata->ompt_task_info.frame.exit_runtime_frame = frame_address;
  }
  if (ompt_enabled.ompt_callback_task_create) {
    ompt_task_info_t *parent_info = &(current_task->ompt_task_info);
    ompt_data_t task_data = ompt_data_none;
    ompt_callbacks.ompt_callback(ompt_callback_task_create)(
        &(parent_info->task_data), &(parent_info->frame),
        &(taskdata->ompt_task_info.task_data),
        ompt_task_explicit | TASK_TYPE_DETAILS_FORMAT(taskdata), 0,
        return_address);
  }
  __ompt_task_start(task, current_task, gtid);

  KA_TRACE(10, ("__kmpc_omp_task_begin_if0(exit): T#%d loc=%p task=%p,\n", gtid,
                loc_ref, taskdata));
}
#endif // OMPT_SUPPORT

// __kmpc_omp_task_begin_if0: report that a given serialized task has started
// execution
//
// loc_ref: source location information; points to beginning of task block.
// gtid: global thread number.
// task: task thunk for the started task.
void __kmpc_omp_task_begin_if0(ident_t *loc_ref, kmp_int32 gtid,
                               kmp_task_t *task) {
#if OMPT_SUPPORT
  if (UNLIKELY(ompt_enabled.enabled)) {
    __ompt_enabled_task_begin_if0(loc_ref, gtid, task,
      OMPT_GET_FRAME_ADDRESS(1), OMPT_GET_RETURN_ADDRESS(0));
    return;
  }
#endif
  kmp_taskdata_t *taskdata = KMP_TASK_TO_TASKDATA(task);
  kmp_taskdata_t *current_task = __kmp_threads[gtid]->th.th_current_task;

  KA_TRACE(10, ("__kmpc_omp_task_begin_if0(enter): T#%d loc=%p task=%p "
                "current_task=%p\n",
                gtid, loc_ref, taskdata, current_task));

  if (taskdata->td_flags.tiedness == TASK_UNTIED) {
    // untied task needs to increment counter so that the task structure is not
    // freed prematurely
    kmp_int32 counter = 1 + KMP_TEST_THEN_INC32(&taskdata->td_untied_count);
    KA_TRACE(20, ("__kmpc_omp_task_begin_if0: T#%d untied_count (%d) "
                  "incremented for task %p\n",
                  gtid, counter, taskdata));
  }

  taskdata->td_flags.task_serial =
      1; // Execute this task immediately, not deferred.

  __kmp_task_start(gtid, task, current_task);

  KA_TRACE(10, ("__kmpc_omp_task_begin_if0(exit): T#%d loc=%p task=%p,\n", gtid,
                loc_ref, taskdata));

  return;
}

#ifdef TASK_UNUSED
// __kmpc_omp_task_begin: report that a given task has started execution
// NEVER GENERATED BY COMPILER, DEPRECATED!!!
void __kmpc_omp_task_begin(ident_t *loc_ref, kmp_int32 gtid, kmp_task_t *task) {
  kmp_taskdata_t *current_task = __kmp_threads[gtid]->th.th_current_task;

  KA_TRACE(
      10,
      ("__kmpc_omp_task_begin(enter): T#%d loc=%p task=%p current_task=%p\n",
       gtid, loc_ref, KMP_TASK_TO_TASKDATA(task), current_task));

  __kmp_task_start(gtid, task, current_task);

  KA_TRACE(10, ("__kmpc_omp_task_begin(exit): T#%d loc=%p task=%p,\n", gtid,
                loc_ref, KMP_TASK_TO_TASKDATA(task)));
  return;
}
#endif // TASK_UNUSED

// __kmp_free_task: free the current task space and the space for shareds
//
// gtid: Global thread ID of calling thread
// taskdata: task to free
// thread: thread data structure of caller
static void __kmp_free_task(kmp_int32 gtid, kmp_taskdata_t *taskdata,
                            kmp_info_t *thread) {
  KA_TRACE(30, ("__kmp_free_task: T#%d freeing data from task %p\n", gtid,
                taskdata));

  // Check to make sure all flags and counters have the correct values
  KMP_DEBUG_ASSERT(taskdata->td_flags.tasktype == TASK_EXPLICIT);
  KMP_DEBUG_ASSERT(taskdata->td_flags.executing == 0);
  KMP_DEBUG_ASSERT(taskdata->td_flags.complete == 1);
  KMP_DEBUG_ASSERT(taskdata->td_flags.freed == 0);
  KMP_DEBUG_ASSERT(TCR_4(taskdata->td_allocated_child_tasks) == 0 ||
                   taskdata->td_flags.task_serial == 1);
  KMP_DEBUG_ASSERT(TCR_4(taskdata->td_incomplete_child_tasks) == 0);

  taskdata->td_flags.freed = 1;
  ANNOTATE_HAPPENS_BEFORE(taskdata);
// deallocate the taskdata and shared variable blocks associated with this task
#if USE_FAST_MEMORY
// #if KMP_USE_TASK_AFFINITY
//   if(enable_numa_aware_stealing){
//     //numa_free(taskdata, taskdata->td_size_alloc);
//   }else{
//     __kmp_fast_free(thread, taskdata);
//   }
// #else
  __kmp_fast_free(thread, taskdata);
// #endif
#else /* ! USE_FAST_MEMORY */
  __kmp_thread_free(thread, taskdata);
#endif

  KA_TRACE(20, ("__kmp_free_task: T#%d freed task %p\n", gtid, taskdata));
}

// __kmp_free_task_and_ancestors: free the current task and ancestors without
// children
//
// gtid: Global thread ID of calling thread
// taskdata: task to free
// thread: thread data structure of caller
static void __kmp_free_task_and_ancestors(kmp_int32 gtid,
                                          kmp_taskdata_t *taskdata,
                                          kmp_info_t *thread) {
#if OMP_45_ENABLED
  // Proxy tasks must always be allowed to free their parents
  // because they can be run in background even in serial mode.
  kmp_int32 team_serial =
      (taskdata->td_flags.team_serial || taskdata->td_flags.tasking_ser) &&
      !taskdata->td_flags.proxy;
#else
  kmp_int32 team_serial =
      taskdata->td_flags.team_serial || taskdata->td_flags.tasking_ser;
#endif
  KMP_DEBUG_ASSERT(taskdata->td_flags.tasktype == TASK_EXPLICIT);

  kmp_int32 children =
      KMP_TEST_THEN_DEC32(&taskdata->td_allocated_child_tasks) - 1;
  KMP_DEBUG_ASSERT(children >= 0);

  // Now, go up the ancestor tree to see if any ancestors can now be freed.
  while (children == 0) {
    kmp_taskdata_t *parent_taskdata = taskdata->td_parent;

    KA_TRACE(20, ("__kmp_free_task_and_ancestors(enter): T#%d task %p complete "
                  "and freeing itself\n",
                  gtid, taskdata));

    // --- Deallocate my ancestor task ---
    __kmp_free_task(gtid, taskdata, thread);

    taskdata = parent_taskdata;

    // Stop checking ancestors at implicit task instead of walking up ancestor
    // tree to avoid premature deallocation of ancestors.
    if (team_serial || taskdata->td_flags.tasktype == TASK_IMPLICIT)
      return;

    // Predecrement simulated by "- 1" calculation
    children = KMP_TEST_THEN_DEC32(&taskdata->td_allocated_child_tasks) - 1;
    KMP_DEBUG_ASSERT(children >= 0);
  }

  KA_TRACE(
      20, ("__kmp_free_task_and_ancestors(exit): T#%d task %p has %d children; "
           "not freeing it yet\n",
           gtid, taskdata, children));
}

// __kmp_task_finish: bookkeeping to do when a task finishes execution
//
// gtid: global thread ID for calling thread
// task: task to be finished
// resumed_task: task to be resumed.  (may be NULL if task is serialized)
static void __kmp_task_finish(kmp_int32 gtid, kmp_task_t *task,
                              kmp_taskdata_t *resumed_task) {
  kmp_taskdata_t *taskdata = KMP_TASK_TO_TASKDATA(task);
  kmp_info_t *thread = __kmp_threads[gtid];
  kmp_task_team_t *task_team =
      thread->th.th_task_team; // might be NULL for serial teams...
  kmp_int32 children = 0;

  KA_TRACE(10, ("__kmp_task_finish(enter): T#%d finishing task %p and resuming "
                "task %p\n",
                gtid, taskdata, resumed_task));

  KMP_DEBUG_ASSERT(taskdata->td_flags.tasktype == TASK_EXPLICIT);

#if KMP_USE_TASK_AFFINITY 
  if (taskdata->naffin > 0)
    free(taskdata->affinity_info);
#endif
#if KMP_USE_TASK_AFFINITY && KMP_TASK_AFFINITY_MEASURE_TIME
  stop_task_execution_measurement(taskdata);
  finish_task_execution_measurement(taskdata, thread);
//#if KMP_TASK_AFFINITY_NEW_MEMORY_ALLOC
//#endif /* KMP_TASK_AFFINITY_NEW_MEMORY_ALLOC */
#endif

// Pop task from stack if tied
#ifdef BUILD_TIED_TASK_STACK
  if (taskdata->td_flags.tiedness == TASK_TIED) {
    __kmp_pop_task_stack(gtid, thread, taskdata);
  }
#endif /* BUILD_TIED_TASK_STACK */

  if (taskdata->td_flags.tiedness == TASK_UNTIED) {
    // untied task needs to check the counter so that the task structure is not
    // freed prematurely
    kmp_int32 counter = KMP_TEST_THEN_DEC32(&taskdata->td_untied_count) - 1;
    KA_TRACE(
        20,
        ("__kmp_task_finish: T#%d untied_count (%d) decremented for task %p\n",
         gtid, counter, taskdata));
    if (counter > 0) {
      // untied task is not done, to be continued possibly by other thread, do
      // not free it now
      if (resumed_task == NULL) {
        KMP_DEBUG_ASSERT(taskdata->td_flags.task_serial);
        resumed_task = taskdata->td_parent; // In a serialized task, the resumed
        // task is the parent
      }
      thread->th.th_current_task = resumed_task; // restore current_task
      resumed_task->td_flags.executing = 1; // resume previous task
      KA_TRACE(10, ("__kmp_task_finish(exit): T#%d partially done task %p, "
                    "resuming task %p\n",
                    gtid, taskdata, resumed_task));
      return;
    }
  }

  KMP_DEBUG_ASSERT(taskdata->td_flags.complete == 0);
  taskdata->td_flags.complete = 1; // mark the task as completed
  KMP_DEBUG_ASSERT(taskdata->td_flags.started == 1);
  KMP_DEBUG_ASSERT(taskdata->td_flags.freed == 0);

  // Only need to keep track of count if team parallel and tasking not
  // serialized
  if (!(taskdata->td_flags.team_serial || taskdata->td_flags.tasking_ser)) {
    // Predecrement simulated by "- 1" calculation
    children =
        KMP_TEST_THEN_DEC32(&taskdata->td_parent->td_incomplete_child_tasks) -
        1;
    KMP_DEBUG_ASSERT(children >= 0);
#if OMP_40_ENABLED
    if (taskdata->td_taskgroup){
      int tmp_count = KMP_TEST_THEN_DEC32((kmp_int32 *)(&taskdata->td_taskgroup->count)) -1;
      //fprintf(stderr, "%f __kmp_task_finish: T#%d decr taskgroup->count to %d\n", get_wall_time2(), gtid, tmp_count);
    }
#if OMP_45_ENABLED
  }
  // if we found proxy tasks there could exist a dependency chain
  // with the proxy task as origin
  if (!(taskdata->td_flags.team_serial || taskdata->td_flags.tasking_ser) ||
      (task_team && task_team->tt.tt_found_proxy_tasks)) {
#endif
    __kmp_release_deps(gtid, taskdata);
#endif
  }

  // td_flags.executing must be marked as 0 after __kmp_release_deps has been
  // called. Othertwise, if a task is executed immediately from the release_deps
  // code, the flag will be reset to 1 again by this same function
  KMP_DEBUG_ASSERT(taskdata->td_flags.executing == 1);
  taskdata->td_flags.executing = 0; // suspend the finishing task

  KA_TRACE(
      20, ("__kmp_task_finish: T#%d finished task %p, %d incomplete children\n",
           gtid, taskdata, children));

#if OMP_40_ENABLED
  /* If the tasks' destructor thunk flag has been set, we need to invoke the
     destructor thunk that has been generated by the compiler. The code is
     placed here, since at this point other tasks might have been released
     hence overlapping the destructor invokations with some other work in the
     released tasks.  The OpenMP spec is not specific on when the destructors
     are invoked, so we should be free to choose. */
  if (taskdata->td_flags.destructors_thunk) {
    kmp_routine_entry_t destr_thunk = task->data1.destructors;
    KMP_ASSERT(destr_thunk);
    destr_thunk(gtid, task);
  }
#endif // OMP_40_ENABLED

  // bookkeeping for resuming task:
  // GEH - note tasking_ser => task_serial
  KMP_DEBUG_ASSERT(
      (taskdata->td_flags.tasking_ser || taskdata->td_flags.task_serial) ==
      taskdata->td_flags.task_serial);
  if (taskdata->td_flags.task_serial) {
    if (resumed_task == NULL) {
      resumed_task = taskdata->td_parent; // In a serialized task, the resumed
      // task is the parent
    } else
#if OMP_45_ENABLED
        if (!(task_team && task_team->tt.tt_found_proxy_tasks))
#endif
    {
      // verify resumed task passed in points to parent
      //suche hier
      KMP_DEBUG_ASSERT(resumed_task == taskdata->td_parent);
    }
  } else {
    KMP_DEBUG_ASSERT(resumed_task !=
                     NULL); // verify that resumed task is passed as arguemnt
  }

  // Free this task and then ancestor tasks if they have no children.
  // Restore th_current_task first as suggested by John:
  // johnmc: if an asynchronous inquiry peers into the runtime system
  // it doesn't see the freed task as the current task.
  thread->th.th_current_task = resumed_task;
  __kmp_free_task_and_ancestors(gtid, taskdata, thread);

  // TODO: GEH - make sure root team implicit task is initialized properly.
  // KMP_DEBUG_ASSERT( resumed_task->td_flags.executing == 0 );
  resumed_task->td_flags.executing = 1; // resume previous task
  KA_TRACE(
      10, ("__kmp_task_finish(exit): T#%d finished task %p, resuming task %p\n",
           gtid, taskdata, resumed_task));

  return;
}

#if OMPT_SUPPORT
OMPT_NOINLINE
static void __ompt_enabled_task_complete_if0(ident_t *loc_ref, kmp_int32 gtid,
                                             kmp_task_t *task) {
  KA_TRACE(10, ("__kmpc_omp_task_complete_if0(enter): T#%d loc=%p task=%p\n",
                gtid, loc_ref, KMP_TASK_TO_TASKDATA(task)));

  // this routine will provide task to resume
  __kmp_task_finish(gtid, task, NULL);

  KA_TRACE(10, ("__kmpc_omp_task_complete_if0(exit): T#%d loc=%p task=%p\n",
                gtid, loc_ref, KMP_TASK_TO_TASKDATA(task)));

  __ompt_task_finish(task, NULL);
  ompt_frame_t *ompt_frame;
  __ompt_get_task_info_internal(0, NULL, NULL, &ompt_frame, NULL, NULL);
  ompt_frame->reenter_runtime_frame = NULL;

  return;
}
#endif

// __kmpc_omp_task_complete_if0: report that a task has completed execution
//
// loc_ref: source location information; points to end of task block.
// gtid: global thread number.
// task: task thunk for the completed task.
void __kmpc_omp_task_complete_if0(ident_t *loc_ref, kmp_int32 gtid,
                                  kmp_task_t *task) {
#if OMPT_SUPPORT
  if (UNLIKELY(ompt_enabled.enabled)) {
    __ompt_enabled_task_complete_if0(loc_ref, gtid, task);
    return;
  }
#endif
  KA_TRACE(10, ("__kmpc_omp_task_complete_if0(enter): T#%d loc=%p task=%p\n",
                gtid, loc_ref, KMP_TASK_TO_TASKDATA(task)));

  // this routine will provide task to resume
  __kmp_task_finish(gtid, task, NULL);

  KA_TRACE(10, ("__kmpc_omp_task_complete_if0(exit): T#%d loc=%p task=%p\n",
                gtid, loc_ref, KMP_TASK_TO_TASKDATA(task)));

  return;
}

#ifdef TASK_UNUSED
// __kmpc_omp_task_complete: report that a task has completed execution
// NEVER GENERATED BY COMPILER, DEPRECATED!!!
void __kmpc_omp_task_complete(ident_t *loc_ref, kmp_int32 gtid,
                              kmp_task_t *task) {
  KA_TRACE(10, ("__kmpc_omp_task_complete(enter): T#%d loc=%p task=%p\n", gtid,
                loc_ref, KMP_TASK_TO_TASKDATA(task)));

  __kmp_task_finish(gtid, task, NULL); // Not sure how to find task to resume

  KA_TRACE(10, ("__kmpc_omp_task_complete(exit): T#%d loc=%p task=%p\n", gtid,
                loc_ref, KMP_TASK_TO_TASKDATA(task)));
  return;
}
#endif // TASK_UNUSED

// __kmp_init_implicit_task: Initialize the appropriate fields in the implicit
// task for a given thread
//
// loc_ref:  reference to source location of parallel region
// this_thr:  thread data structure corresponding to implicit task
// team: team for this_thr
// tid: thread id of given thread within team
// set_curr_task: TRUE if need to push current task to thread
// NOTE: Routine does not set up the implicit task ICVS.  This is assumed to
// have already been done elsewhere.
// TODO: Get better loc_ref.  Value passed in may be NULL
void __kmp_init_implicit_task(ident_t *loc_ref, kmp_info_t *this_thr,
                              kmp_team_t *team, int tid, int set_curr_task) {
  kmp_taskdata_t *task = &team->t.t_implicit_task_taskdata[tid];

  KF_TRACE(
      10,
      ("__kmp_init_implicit_task(enter): T#:%d team=%p task=%p, reinit=%s\n",
       tid, team, task, set_curr_task ? "TRUE" : "FALSE"));

  task->td_task_id = KMP_GEN_TASK_ID();
  task->td_team = team;
  //    task->td_parent   = NULL;  // fix for CQ230101 (broken parent task info
  //    in debugger)
  task->td_ident = loc_ref;
  task->td_taskwait_ident = NULL;
  task->td_taskwait_counter = 0;
  task->td_taskwait_thread = 0;

  task->td_flags.tiedness = TASK_TIED;
  task->td_flags.tasktype = TASK_IMPLICIT;
#if OMP_45_ENABLED
  task->td_flags.proxy = TASK_FULL;
#endif

  // All implicit tasks are executed immediately, not deferred
  task->td_flags.task_serial = 1;
  task->td_flags.tasking_ser = (__kmp_tasking_mode == tskm_immediate_exec);
  task->td_flags.team_serial = (team->t.t_serialized) ? 1 : 0;

  task->td_flags.started = 1;
  task->td_flags.executing = 1;
  task->td_flags.complete = 0;
  task->td_flags.freed = 0;

#if OMP_40_ENABLED
  task->td_depnode = NULL;
#endif

  if (set_curr_task) { // only do this init first time thread is created
    task->td_incomplete_child_tasks = 0;
    // Not used: don't need to deallocate implicit task
    task->td_allocated_child_tasks = 0;
#if OMP_40_ENABLED
    task->td_taskgroup = NULL; // An implicit task does not have taskgroup
    task->td_dephash = NULL;
#endif
    __kmp_push_current_task_to_thread(this_thr, team, tid);
  } else {
    KMP_DEBUG_ASSERT(task->td_incomplete_child_tasks == 0);
    KMP_DEBUG_ASSERT(task->td_allocated_child_tasks == 0);
  }

#if OMPT_SUPPORT
  if (__builtin_expect(ompt_enabled.enabled,0)) __ompt_task_init(task, tid);
#endif

  KF_TRACE(10, ("__kmp_init_implicit_task(exit): T#:%d team=%p task=%p\n", tid,
                team, task));
}

// __kmp_finish_implicit_task: Release resources associated to implicit tasks
// at the end of parallel regions. Some resources are kept for reuse in the next
// parallel region.
//
// thread:  thread data structure corresponding to implicit task
void __kmp_finish_implicit_task(kmp_info_t *thread) {
  kmp_taskdata_t *task = thread->th.th_current_task;
  if (task->td_dephash)
    __kmp_dephash_free_entries(thread, task->td_dephash);
}

// __kmp_free_implicit_task: Release resources associated to implicit tasks
// when these are destroyed regions
//
// thread:  thread data structure corresponding to implicit task
void __kmp_free_implicit_task(kmp_info_t *thread) {
  kmp_taskdata_t *task = thread->th.th_current_task;
  if (task->td_dephash)
    __kmp_dephash_free(thread, task->td_dephash);
  task->td_dephash = NULL;
}

// Round up a size to a power of two specified by val: Used to insert padding
// between structures co-allocated using a single malloc() call
static size_t __kmp_round_up_to_val(size_t size, size_t val) {
  if (size & (val - 1)) {
    size &= ~(val - 1);
    if (size <= KMP_SIZE_T_MAX - val) {
      size += val; // Round up if there is no overflow.
    }; // if
  }; // if
  return size;
} // __kmp_round_up_to_va

// __kmp_task_alloc: Allocate the taskdata and task data structures for a task
//
// loc_ref: source location information
// gtid: global thread number.
// flags: include tiedness & task type (explicit vs. implicit) of the ''new''
// task encountered. Converted from kmp_int32 to kmp_tasking_flags_t in routine.
// sizeof_kmp_task_t:  Size in bytes of kmp_task_t data structure including
// private vars accessed in task.
// sizeof_shareds:  Size in bytes of array of pointers to shared vars accessed
// in task.
// task_entry: Pointer to task code entry point generated by compiler.
// returns: a pointer to the allocated kmp_task_t structure (task).
kmp_task_t *__kmp_task_alloc(ident_t *loc_ref, kmp_int32 gtid,
                             kmp_tasking_flags_t *flags,
                             size_t sizeof_kmp_task_t, size_t sizeof_shareds,
                             kmp_routine_entry_t task_entry) {
  kmp_task_t *task;
  kmp_taskdata_t *taskdata;
  kmp_info_t *thread = __kmp_threads[gtid];
  kmp_team_t *team = thread->th.th_team;
  kmp_taskdata_t *parent_task = thread->th.th_current_task;
  size_t shareds_offset;

  KA_TRACE(10, ("__kmp_task_alloc(enter): T#%d loc=%p, flags=(0x%x) "
                "sizeof_task=%ld sizeof_shared=%ld entry=%p\n",
                gtid, loc_ref, *((kmp_int32 *)flags), sizeof_kmp_task_t,
                sizeof_shareds, task_entry));

  if (parent_task->td_flags.final) {
    if (flags->merged_if0) {
    }
    flags->final = 1;
  }

#if OMP_45_ENABLED
  if (flags->proxy == TASK_PROXY) {
    flags->tiedness = TASK_UNTIED;
    flags->merged_if0 = 1;

    /* are we running in a sequential parallel or tskm_immediate_exec... we need
       tasking support enabled */
    if ((thread->th.th_task_team) == NULL) {
      /* This should only happen if the team is serialized
          setup a task team and propagate it to the thread */
      KMP_DEBUG_ASSERT(team->t.t_serialized);
      KA_TRACE(30,
               ("T#%d creating task team in __kmp_task_alloc for proxy task\n",
                gtid));
      __kmp_task_team_setup(
          thread, team,
          1); // 1 indicates setup the current team regardless of nthreads
      thread->th.th_task_team = team->t.t_task_team[thread->th.th_task_state];
    }
    kmp_task_team_t *task_team = thread->th.th_task_team;

    /* tasking must be enabled now as the task might not be pushed */
    if (!KMP_TASKING_ENABLED(task_team)) {
      KA_TRACE(
          30,
          ("T#%d enabling tasking in __kmp_task_alloc for proxy task\n", gtid));
      __kmp_enable_tasking(task_team, thread);
      kmp_int32 tid = thread->th.th_info.ds.ds_tid;
      kmp_thread_data_t *thread_data = &task_team->tt.tt_threads_data[tid];
      // No lock needed since only owner can allocate
      if (thread_data->td.td_deque == NULL) {
        __kmp_alloc_task_deque(thread, thread_data);
      }
    }

    if (task_team->tt.tt_found_proxy_tasks == FALSE)
      TCW_4(task_team->tt.tt_found_proxy_tasks, TRUE);
  }
#endif

  // Calculate shared structure offset including padding after kmp_task_t struct
  // to align pointers in shared struct
  shareds_offset = sizeof(kmp_taskdata_t) + sizeof_kmp_task_t;
  shareds_offset = __kmp_round_up_to_val(shareds_offset, sizeof(void *));

  // Allocate a kmp_taskdata_t block and a kmp_task_t block.
  KA_TRACE(30, ("__kmp_task_alloc: T#%d First malloc size: %ld\n", gtid,
                shareds_offset));
  KA_TRACE(30, ("__kmp_task_alloc: T#%d Second malloc size: %ld\n", gtid,
                sizeof_shareds));

// Avoid double allocation here by combining shareds with taskdata
#if USE_FAST_MEMORY
// #if KMP_USE_TASK_AFFINITY
// if(enable_numa_aware_stealing)
// {
// 	if(thread->th.th_task_affinity_data != NULL)
// 	{
// 		int current_data_domain = -1;
// 		int ret_code = move_pages(0 /*self memory */, 1, &thread->th.th_task_affinity_data, NULL, &current_data_domain, 0);
// 		if(ret_code == 0 && current_data_domain >= 0 && current_data_domain != thread->th.th_task_aff_my_domain_nr)
// 		{
// 			// get buffer from different thread that is pinned to NUMA domain
// 			srand(time(NULL));
// 			int idx_rand = rand() % numa_domain_size[current_data_domain];
// 			int tmp_gtid = map_threads_in_numa_domain[current_data_domain][idx_rand];

// 			kmp_info_t *tmp_thread = __kmp_threads[tmp_gtid];
// 			// fprintf(stderr, "__kmp_task_alloc: T#%d allocating taskdata on NUMA node %d using thread T#%d\n", gtid, current_data_domain, tmp_gtid);

// 			// taskdata = (kmp_taskdata_t *)numa_alloc_onnode(shareds_offset + sizeof_shareds, current_data_domain);
// 			taskdata = (kmp_taskdata_t *)__kmp_fast_allocate(tmp_thread, shareds_offset + sizeof_shareds);
// 		} else {
// 			taskdata = (kmp_taskdata_t *)__kmp_fast_allocate(thread, shareds_offset + sizeof_shareds);
// 		}
// 	} else {
// 		taskdata = (kmp_taskdata_t *)__kmp_fast_allocate(thread, shareds_offset + sizeof_shareds);
// 	}
// } else {
// 	taskdata = (kmp_taskdata_t *)__kmp_fast_allocate(thread, shareds_offset + sizeof_shareds);
// }
// #else
  taskdata = (kmp_taskdata_t *)__kmp_fast_allocate(thread, shareds_offset + sizeof_shareds);
// #endif
#else /* ! USE_FAST_MEMORY */
  taskdata = (kmp_taskdata_t *)__kmp_thread_malloc(thread, shareds_offset +
                                                               sizeof_shareds);
#endif /* USE_FAST_MEMORY */
  ANNOTATE_HAPPENS_AFTER(taskdata);

  task = KMP_TASKDATA_TO_TASK(taskdata);

// Make sure task & taskdata are aligned appropriately
#if KMP_ARCH_X86 || KMP_ARCH_PPC64 || !KMP_HAVE_QUAD
  KMP_DEBUG_ASSERT((((kmp_uintptr_t)taskdata) & (sizeof(double) - 1)) == 0);
  KMP_DEBUG_ASSERT((((kmp_uintptr_t)task) & (sizeof(double) - 1)) == 0);
#else
  KMP_DEBUG_ASSERT((((kmp_uintptr_t)taskdata) & (sizeof(_Quad) - 1)) == 0);
  KMP_DEBUG_ASSERT((((kmp_uintptr_t)task) & (sizeof(_Quad) - 1)) == 0);
#endif
  if (sizeof_shareds > 0) {
    // Avoid double allocation here by combining shareds with taskdata
    task->shareds = &((char *)taskdata)[shareds_offset];
    // Make sure shareds struct is aligned to pointer size
    KMP_DEBUG_ASSERT((((kmp_uintptr_t)task->shareds) & (sizeof(void *) - 1)) ==
                     0);
  } else {
    task->shareds = NULL;
  }
  task->routine = task_entry;
  task->part_id = 0; // AC: Always start with 0 part id

  taskdata->td_task_id = KMP_GEN_TASK_ID();
  taskdata->td_team = team;
  taskdata->td_alloc_thread = thread;
  taskdata->td_parent = parent_task;
  taskdata->td_level = parent_task->td_level + 1; // increment nesting level
  taskdata->td_untied_count = 0;
  taskdata->td_ident = loc_ref;
  taskdata->td_taskwait_ident = NULL;
  taskdata->td_taskwait_counter = 0;
  taskdata->td_taskwait_thread = 0;
  KMP_DEBUG_ASSERT(taskdata->td_parent != NULL);
#if OMP_45_ENABLED
  // avoid copying icvs for proxy tasks
  if (flags->proxy == TASK_FULL)
#endif
    copy_icvs(&taskdata->td_icvs, &taskdata->td_parent->td_icvs);

  taskdata->td_flags.tiedness = flags->tiedness;
  taskdata->td_flags.final = flags->final;
  taskdata->td_flags.merged_if0 = flags->merged_if0;
#if OMP_40_ENABLED
  taskdata->td_flags.destructors_thunk = flags->destructors_thunk;
#endif // OMP_40_ENABLED
#if OMP_45_ENABLED
  taskdata->td_flags.proxy = flags->proxy;
  taskdata->td_task_team = thread->th.th_task_team;
  taskdata->td_size_alloc = shareds_offset + sizeof_shareds;
#endif
  taskdata->td_flags.tasktype = TASK_EXPLICIT;

  // GEH - TODO: fix this to copy parent task's value of tasking_ser flag
  taskdata->td_flags.tasking_ser = (__kmp_tasking_mode == tskm_immediate_exec);

  // GEH - TODO: fix this to copy parent task's value of team_serial flag
  taskdata->td_flags.team_serial = (team->t.t_serialized) ? 1 : 0;

  // GEH - Note we serialize the task if the team is serialized to make sure
  // implicit parallel region tasks are not left until program termination to
  // execute. Also, it helps locality to execute immediately.

  taskdata->td_flags.task_serial =
      (parent_task->td_flags.final || taskdata->td_flags.team_serial ||
       taskdata->td_flags.tasking_ser);

  taskdata->td_flags.started = 0;
  taskdata->td_flags.executing = 0;
  taskdata->td_flags.complete = 0;
  taskdata->td_flags.freed = 0;

  taskdata->td_flags.native = flags->native;

  taskdata->td_incomplete_child_tasks = 0;
  taskdata->td_allocated_child_tasks = 1; // start at one because counts current
// task and children
#if OMP_40_ENABLED
  taskdata->td_taskgroup =
      parent_task->td_taskgroup; // task inherits taskgroup from the parent task
  taskdata->td_dephash = NULL;
  taskdata->td_depnode = NULL;
#endif

// Only need to keep track of child task counts if team parallel and tasking not
// serialized or if it is a proxy task
#if OMP_45_ENABLED
  if (flags->proxy == TASK_PROXY ||
      !(taskdata->td_flags.team_serial || taskdata->td_flags.tasking_ser))
#else
  if (!(taskdata->td_flags.team_serial || taskdata->td_flags.tasking_ser))
#endif
  {
    KMP_TEST_THEN_INC32(&parent_task->td_incomplete_child_tasks);
#if OMP_40_ENABLED
    if (parent_task->td_taskgroup)
      {
      int tmp_count = KMP_TEST_THEN_INC32((kmp_int32 *)(&parent_task->td_taskgroup->count)) + 1;
      //fprintf(stderr, "%f __kmp_task_alloc: T#%d incr taskgroup->count to %d\n", get_wall_time2(), gtid, tmp_count);
      }
#endif
    // Only need to keep track of allocated child tasks for explicit tasks since
    // implicit not deallocated
    if (taskdata->td_parent->td_flags.tasktype == TASK_EXPLICIT) {
      KMP_TEST_THEN_INC32(&taskdata->td_parent->td_allocated_child_tasks);
    }
  }

#if KMP_USE_TASK_AFFINITY
  taskdata->affinity_info = NULL;
  taskdata->naffin = 0;

  // remember where task has been executed and domain where data is located
  taskdata->td_task_affinity_scheduled_thread_set = false;
  taskdata->td_task_affinity_scheduled_thread = -1;
  taskdata->td_task_affinity_data_domain = -1;

  taskdata->td_ts_task_execution = -1;
  taskdata->td_ts_task_execution_current_sum = 0.0;
#endif

  KA_TRACE(20, ("__kmp_task_alloc(exit): T#%d created task %p parent=%p\n",
                gtid, taskdata, taskdata->td_parent));
  ANNOTATE_HAPPENS_BEFORE(task);

#if OMPT_SUPPORT
  if (__builtin_expect(ompt_enabled.enabled,0)) __ompt_task_init(taskdata, gtid);
#endif

  return task;
}

kmp_task_t *__kmpc_omp_task_alloc(ident_t *loc_ref, kmp_int32 gtid,
                                  kmp_int32 flags, size_t sizeof_kmp_task_t,
                                  size_t sizeof_shareds,
                                  kmp_routine_entry_t task_entry) {
  kmp_task_t *retval;
  kmp_tasking_flags_t *input_flags = (kmp_tasking_flags_t *)&flags;

  input_flags->native = FALSE;
// __kmp_task_alloc() sets up all other runtime flags

#if OMP_45_ENABLED
  KA_TRACE(10, ("__kmpc_omp_task_alloc(enter): T#%d loc=%p, flags=(%s %s) "
                "sizeof_task=%ld sizeof_shared=%ld entry=%p\n",
                gtid, loc_ref, input_flags->tiedness ? "tied  " : "untied",
                input_flags->proxy ? "proxy" : "", sizeof_kmp_task_t,
                sizeof_shareds, task_entry));
#else
  KA_TRACE(10, ("__kmpc_omp_task_alloc(enter): T#%d loc=%p, flags=(%s) "
                "sizeof_task=%ld sizeof_shared=%ld entry=%p\n",
                gtid, loc_ref, input_flags->tiedness ? "tied  " : "untied",
                sizeof_kmp_task_t, sizeof_shareds, task_entry));
#endif

  retval = __kmp_task_alloc(loc_ref, gtid, input_flags, sizeof_kmp_task_t,
                            sizeof_shareds, task_entry);

  KA_TRACE(20, ("__kmpc_omp_task_alloc(exit): T#%d retval %p\n", gtid, retval));
  
  //suche

  kmp_info_t *thread = __kmp_threads[gtid];
  __kmpc_omp_reg_task_with_affinity(loc_ref, gtid, retval, thread->th.naffin, thread->th.th_task_affinity_data);
  return retval;
}

//  __kmp_invoke_task: invoke the specified task
//
// gtid: global thread ID of caller
// task: the task to invoke
// current_task: the task to resume after task invokation
static void __kmp_invoke_task(kmp_int32 gtid, kmp_task_t *task,
                              kmp_taskdata_t *current_task) {
  kmp_taskdata_t *taskdata = KMP_TASK_TO_TASKDATA(task);
  kmp_uint64 cur_time;
#if OMP_40_ENABLED
  int discard = 0 /* false */;
#endif
  // KA_TRACE(
  //     30, ("__kmp_invoke_task(enter): T#%d invoking task %p, current_task=%p\n",
  //          gtid, taskdata, current_task));
  KA_TRACE(
      30, ("__kmp_invoke_task(enter): T#%d invoking task %p lvl %d, current_task=%p lvl %d\n",
           gtid, taskdata, taskdata->td_level, current_task, current_task->td_level));
  KMP_DEBUG_ASSERT(task);
#if OMP_45_ENABLED
  if (taskdata->td_flags.proxy == TASK_PROXY &&
      taskdata->td_flags.complete == 1) {
    // This is a proxy task that was already completed but it needs to run
    // its bottom-half finish
    KA_TRACE(
        30,
        ("__kmp_invoke_task: T#%d running bottom finish for proxy task %p\n",
         gtid, taskdata));

    __kmp_bottom_half_finish_proxy(gtid, task);

    KA_TRACE(30, ("__kmp_invoke_task(exit): T#%d completed bottom finish for "
                  "proxy task %p, resuming task %p\n",
                  gtid, taskdata, current_task));

    return;
  }
#endif

#if USE_ITT_BUILD && USE_ITT_NOTIFY
  if (__kmp_forkjoin_frames_mode == 3) {
    // Get the current time stamp to measure task execution time to correct
    // barrier imbalance time
    cur_time = __itt_get_timestamp();
  }
#endif

#if OMP_45_ENABLED
  // Proxy tasks are not handled by the runtime
  if (taskdata->td_flags.proxy != TASK_PROXY) {
#endif
    ANNOTATE_HAPPENS_AFTER(task);
    __kmp_task_start(gtid, task, current_task); // OMPT only if not discarded
#if OMP_45_ENABLED
  }
#endif

#if OMPT_SUPPORT
  ompt_thread_info_t oldInfo;
  kmp_info_t *thread;
  if (__builtin_expect(ompt_enabled.enabled,0)) {
    // Store the threads states and restore them after the task
    thread = __kmp_threads[gtid];
    oldInfo = thread->th.ompt_thread_info;
    thread->th.ompt_thread_info.wait_id = 0;
    thread->th.ompt_thread_info.state = (thread->th.th_team_serialized)?omp_state_work_serial:omp_state_work_parallel;
    taskdata->ompt_task_info.frame.exit_runtime_frame =
        OMPT_GET_FRAME_ADDRESS(0);
  }
#endif

#if OMP_40_ENABLED
  // TODO: cancel tasks if the parallel region has also been cancelled
  // TODO: check if this sequence can be hoisted above __kmp_task_start
  // if cancellation has been enabled for this run ...
  if (__kmp_omp_cancellation) {
    kmp_info_t *this_thr = __kmp_threads[gtid];
    kmp_team_t *this_team = this_thr->th.th_team;
    kmp_taskgroup_t *taskgroup = taskdata->td_taskgroup;
    if ((taskgroup && taskgroup->cancel_request) ||
        (this_team->t.t_cancel_request == cancel_parallel)) {
#if OMPT_SUPPORT && OMPT_OPTIONAL
      ompt_data_t *task_data;
      if (__builtin_expect(ompt_enabled.ompt_callback_cancel,0)) {
        __ompt_get_task_info_internal(0, NULL, &task_data, NULL, NULL, NULL);
        ompt_callbacks.ompt_callback(ompt_callback_cancel)(
            task_data,
            ((taskgroup && taskgroup->cancel_request) ? ompt_cancel_taskgroup
                                                      : ompt_cancel_parallel) |
                ompt_cancel_discarded_task,
            NULL);
      }
#endif
      KMP_COUNT_BLOCK(TASK_cancelled);
      // this task belongs to a task group and we need to cancel it
      discard = 1 /* true */;
    }
  }

  // Invoke the task routine and pass in relevant data.
  // Thunks generated by gcc take a different argument list.
  if (!discard) {
#if KMP_STATS_ENABLED
    KMP_COUNT_BLOCK(TASK_executed);
    switch (KMP_GET_THREAD_STATE()) {
    case FORK_JOIN_BARRIER:
      KMP_PUSH_PARTITIONED_TIMER(OMP_task_join_bar);
      break;
    case PLAIN_BARRIER:
      KMP_PUSH_PARTITIONED_TIMER(OMP_task_plain_bar);
      break;
    case TASKYIELD:
      KMP_PUSH_PARTITIONED_TIMER(OMP_task_taskyield);
      break;
    case TASKWAIT:
      KMP_PUSH_PARTITIONED_TIMER(OMP_task_taskwait);
      break;
    case TASKGROUP:
      KMP_PUSH_PARTITIONED_TIMER(OMP_task_taskgroup);
      break;
    default:
      KMP_PUSH_PARTITIONED_TIMER(OMP_task_immediate);
      break;
    }
#endif // KMP_STATS_ENABLED
#endif // OMP_40_ENABLED

// OMPT task begin
#if OMPT_SUPPORT
    if (__builtin_expect(ompt_enabled.enabled,0)) __ompt_task_start(task, current_task, gtid);
#endif
#if KMP_USE_TASK_AFFINITY && KMP_TASK_AFFINITY_MEASURE_TIME
    stop_task_execution_measurement(current_task);
    start_task_execution_measurement(taskdata);
#endif

#ifdef KMP_GOMP_COMPAT
    if (taskdata->td_flags.native) {
      ((void (*)(void *))(*(task->routine)))(task->shareds);
    } else
#endif /* KMP_GOMP_COMPAT */
    {
      (*(task->routine))(gtid, task);
    }
    KMP_POP_PARTITIONED_TIMER();

#if OMPT_SUPPORT
    if (__builtin_expect(ompt_enabled.enabled,0)) __ompt_task_finish(task, current_task);
#endif
#if KMP_USE_TASK_AFFINITY && KMP_TASK_AFFINITY_MEASURE_TIME
    stop_task_execution_measurement(taskdata);
    // do not resume current task here, just after taskwait construct
    // rest is waiting time in barrier
    // start_task_execution_measurement(current_task);
#endif
#if OMP_40_ENABLED
  }
#endif // OMP_40_ENABLED

#if OMPT_SUPPORT
  if (__builtin_expect(ompt_enabled.enabled,0)) {
    thread->th.ompt_thread_info = oldInfo;
    taskdata->ompt_task_info.frame.exit_runtime_frame = NULL;
  }
#endif

#if OMP_45_ENABLED
  // Proxy tasks are not handled by the runtime
  if (taskdata->td_flags.proxy != TASK_PROXY) {
#endif
    ANNOTATE_HAPPENS_BEFORE(taskdata->td_parent);
    __kmp_task_finish(gtid, task, current_task); // OMPT only if not discarded
#if OMP_45_ENABLED
  }
#endif

#if USE_ITT_BUILD && USE_ITT_NOTIFY
  // Barrier imbalance - correct arrive time after the task finished
  if (__kmp_forkjoin_frames_mode == 3) {
    kmp_info_t *this_thr = __kmp_threads[gtid];
    if (this_thr->th.th_bar_arrive_time) {
      this_thr->th.th_bar_arrive_time += (__itt_get_timestamp() - cur_time);
    }
  }
#endif
  KA_TRACE(
      30,
      ("__kmp_invoke_task(exit): T#%d completed task %p, resuming task %p\n",
       gtid, taskdata, current_task));
  return;
}

// __kmpc_omp_task_parts: Schedule a thread-switchable task for execution
//
// loc_ref: location of original task pragma (ignored)
// gtid: Global Thread ID of encountering thread
// new_task: task thunk allocated by __kmp_omp_task_alloc() for the ''new task''
// Returns:
//    TASK_CURRENT_NOT_QUEUED (0) if did not suspend and queue current task to
//    be resumed later.
//    TASK_CURRENT_QUEUED (1) if suspended and queued the current task to be
//    resumed later.
kmp_int32 __kmpc_omp_task_parts(ident_t *loc_ref, kmp_int32 gtid,
                                kmp_task_t *new_task) {
  kmp_taskdata_t *new_taskdata = KMP_TASK_TO_TASKDATA(new_task);

  KA_TRACE(10, ("__kmpc_omp_task_parts(enter): T#%d loc=%p task=%p\n", gtid,
                loc_ref, new_taskdata));
kmp_int32 res;

#if OMPT_SUPPORT
  kmp_taskdata_t *parent;
  if (__builtin_expect(ompt_enabled.enabled,0)) {
    parent = new_taskdata->td_parent;
    //        parent->ompt_task_info.frame.reenter_runtime_frame =
    //            OMPT_GET_FRAME_ADDRESS(1);
    if (ompt_enabled.ompt_callback_task_create) {
      ompt_data_t task_data = ompt_data_none;
      ompt_callbacks.ompt_callback(ompt_callback_task_create)(
          parent ? &(parent->ompt_task_info.task_data) : &task_data,
          parent ? &(parent->ompt_task_info.frame) : NULL,
          &(new_taskdata->ompt_task_info.task_data), ompt_task_explicit, 0,
          OMPT_GET_RETURN_ADDRESS(0));
    }
  }
#endif

  /* Should we execute the new task or queue it? For now, let's just always try
     to queue it.  If the queue fills up, then we'll execute it.  */

  if (__kmp_push_task(gtid, new_task) == TASK_NOT_PUSHED) // if cannot defer
  { // Execute this task immediately
    kmp_taskdata_t *current_task = __kmp_threads[gtid]->th.th_current_task;
    new_taskdata->td_flags.task_serial = 1;
    __kmp_invoke_task(gtid, new_task, current_task);
  }

  KA_TRACE(
      10,
      ("__kmpc_omp_task_parts(exit): T#%d returning TASK_CURRENT_NOT_QUEUED: "
       "loc=%p task=%p, return: TASK_CURRENT_NOT_QUEUED\n",
       gtid, loc_ref, new_taskdata));

  ANNOTATE_HAPPENS_BEFORE(new_task);
#if OMPT_SUPPORT
  if (__builtin_expect(ompt_enabled.enabled,0)) {
    parent->ompt_task_info.frame.reenter_runtime_frame = NULL;
  }
#endif
  return TASK_CURRENT_NOT_QUEUED;
}

#if KMP_USE_TASK_AFFINITY
// __kmp_omp_task_aff: Schedule a non-thread-switchable task for execution on another thread
//
// gtid: Global Thread ID of encountering thread
// new_task:non-thread-switchable task thunk allocated by __kmp_omp_task_alloc()
// serialize_immediate: if TRUE then if the task is executed immediately its
// execution will be serialized
// Returns:
//    TASK_CURRENT_NOT_QUEUED (0) if did not suspend and queue current task to
//    be resumed later.
//    TASK_CURRENT_QUEUED (1) if suspended and queued the current task to be
//    resumed later.
kmp_int32 __kmp_omp_task_aff(kmp_int32 gtid, kmp_int32 target_gtid, kmp_task_t *new_task,
                         bool serialize_immediate) {
  kmp_taskdata_t *new_taskdata = KMP_TASK_TO_TASKDATA(new_task);

  KA_TRACE(10, ("__kmp_omp_task_aff: T#%d Creating task for thread T#%d: %p task_lvl: %d --- parent: %p parent_lvl: %d\n",
    gtid, target_gtid, new_taskdata, new_taskdata->td_level,  new_taskdata->td_parent, new_taskdata->td_parent->td_level));
/* Should we execute the new task or queue it? For now, let's just always try to
   queue it.  If the queue fills up, then we'll execute it.  */

#if OMP_45_ENABLED
  if (new_taskdata->td_flags.proxy == TASK_PROXY ||
      __kmp_push_task_aff(target_gtid, gtid, new_task) == TASK_NOT_PUSHED) // if cannot defer
#else
  if (__kmp_push_task_aff(target_gtid, gtid, new_task) == TASK_NOT_PUSHED) // if cannot defer
#endif
  { // Execute this task immediately
// #if KMP_TASK_AFFINITY_MEASURE_TIME
//     double time2 = get_wall_time2();
// #endif
    KA_TRACE(10, ("__kmp_omp_task_aff: T#%d immediately executing task.\n", gtid));
    kmp_taskdata_t *current_task = __kmp_threads[gtid]->th.th_current_task;
    if (serialize_immediate)
      new_taskdata->td_flags.task_serial = 1;
    __kmp_invoke_task(gtid, new_task, current_task);
// #if KMP_TASK_AFFINITY_MEASURE_TIME
//     time2 = get_wall_time2()-time2;
//     __kmp_threads[gtid]->th.th_sum_time_pushing_inaff += time2;
//     __kmp_threads[gtid]->th.th_num_pushing_inaff++;
// #endif
  }

  ANNOTATE_HAPPENS_BEFORE(new_task);
  return TASK_CURRENT_NOT_QUEUED;
}
#endif

// __kmp_omp_task: Schedule a non-thread-switchable task for execution
//
// gtid: Global Thread ID of encountering thread
// new_task:non-thread-switchable task thunk allocated by __kmp_omp_task_alloc()
// serialize_immediate: if TRUE then if the task is executed immediately its
// execution will be serialized
// Returns:
//    TASK_CURRENT_NOT_QUEUED (0) if did not suspend and queue current task to
//    be resumed later.
//    TASK_CURRENT_QUEUED (1) if suspended and queued the current task to be
//    resumed later.
kmp_int32 __kmp_omp_task(kmp_int32 gtid, kmp_task_t *new_task,
                         bool serialize_immediate) {
  kmp_taskdata_t *new_taskdata = KMP_TASK_TO_TASKDATA(new_task);

  KA_TRACE(10, ("__kmp_omp_task: T#%d Creating task: %p task_lvl: %d --- parent: %p parent_lvl: %d\n",
    gtid, new_taskdata, new_taskdata->td_level,  new_taskdata->td_parent, new_taskdata->td_parent->td_level));
/* Should we execute the new task or queue it? For now, let's just always try to
   queue it.  If the queue fills up, then we'll execute it.  */

#if OMP_45_ENABLED
  if (new_taskdata->td_flags.proxy == TASK_PROXY ||
      __kmp_push_task(gtid, new_task) == TASK_NOT_PUSHED) // if cannot defer
#else
  if (__kmp_push_task(gtid, new_task) == TASK_NOT_PUSHED) // if cannot defer
#endif
  { // Execute this task immediately
    KA_TRACE(10, ("__kmp_omp_task: T#%d immediately executing task.\n", gtid));
    kmp_taskdata_t *current_task = __kmp_threads[gtid]->th.th_current_task;
    if (serialize_immediate)
      new_taskdata->td_flags.task_serial = 1;
    __kmp_invoke_task(gtid, new_task, current_task);
  }

  ANNOTATE_HAPPENS_BEFORE(new_task);
  return TASK_CURRENT_NOT_QUEUED;
}

task_aff_physical_data_location_t inline check_page(kmp_int32 gtid, kmp_info_t *thread, kmp_intptr_t addr){
    KA_TRACE(50,("check_page (enter) T#%d: addr %p\n", gtid, addr));
    #if KMP_TASK_AFFINITY_MEASURE_TIME
            double time2 = get_wall_time2();
    #endif

    int ret=-1, found=0;
    const int page_size = KMP_GET_PAGE_SIZE();
    size_t page_start_address = addr & ~(page_size-1);
    void * page_boundary_pointer = (void *) page_start_address;
    kmp_task_team_t *task_team = thread->th.th_task_team;
    kmp_thread_data_t *threads_data = (kmp_thread_data_t *)TCR_PTR(task_team->tt.tt_threads_data);

    #if KMP_TASK_AFFINITY_USE_DEFAULT_MAP
        auto search = task_aff_addr_map.find(page_start_address);
        found = search != task_aff_addr_map.end();

        //if (search->second.data_domain == -1 || search->second.gtid == -1)
          //found = false;
    #else
        kmp_maphash_entry * cur_entry = __kmp_maphash_find(thread, task_aff_addr_map2, (kmp_intptr_t) page_start_address);
        found = cur_entry->val.gtid != -1 && cur_entry->val.data_domain != -1;
    #endif

    #if KMP_TASK_AFFINITY_ALWAYS_CHECK_PHYSICAL_LOCATION
            found = false;
    #endif

    //if (kmp_affinity_settings.affinity_map_mode == kmp_affinity_map_type_domain)
      //found = false;


    #if KMP_TASK_AFFINITY_MEASURE_TIME
            time2 = get_wall_time2()-time2;
            thread->th.th_sum_time_map_find += time2;
            thread->th.th_num_map_find++;
    #endif

    int current_data_domain=-1; int target_tid=-1; int target_gtid=-1;
    
    if (found){
        #if KMP_TASK_AFFINITY_MEASURE_TIME
            thread->th.th_count_map_found++;
        #endif

        #if KMP_TASK_AFFINITY_USE_DEFAULT_MAP
            return search->second;
        #else
            return cur_entry->val;
        #endif
    } else {
        #if KMP_TASK_AFFINITY_MEASURE_TIME
            thread->th.th_count_map_not_found++;
            time2 = get_wall_time2();
        #endif
        #if KMP_TASK_AFFINITY_MEASURE_TIME
                  time2 = get_wall_time2();
        #endif

        int ret_code = move_pages(0 /*self memory */, 1, &page_boundary_pointer, NULL, &current_data_domain, 0);

        #if KMP_TASK_AFFINITY_MEASURE_TIME
                  time2 = get_wall_time2() - time2;
                  thread->th.th_sum_time_identify_physical_location += time2;
                  thread->th.th_num_identify_physical_location++;
        #endif
        KA_TRACE(60, ("__kmpc_omp_task: T#%d Memory at %p is at numa node\t%d\t(retcode %d)\n", gtid, page_start_address, current_data_domain, ret_code));
        if (ret_code == 0 && current_data_domain >= 0){
            if(kmp_affinity_settings.thread_selection_strategy == kmp_affinity_thread_selection_mode_private
                && current_data_domain == thread->th.th_task_aff_my_domain_nr
                && (kmp_affinity_settings.affinity_map_mode == kmp_affinity_map_type_domain || kmp_affinity_settings.affinity_map_mode == kmp_affinity_map_type_combined)) 
                {
                  target_gtid = gtid;
                  target_tid = __kmp_tid_from_gtid(gtid);
                }

            KA_TRACE(50,("calling init thread: domain %d, team %d, threads %d, tid %d, gtid %d, idx %d\n",current_data_domain, task_team, threads_data, target_tid, target_gtid, threads_data->td.td_idx_in_numa_map));
            kmp_info_t * target_thread = __kmp_task_aff_get_initial_thread_in_numa_domain(current_data_domain, task_team, threads_data, &target_tid, &target_gtid);

            task_aff_physical_data_location_t tmp_result;
            if(target_tid != -1) {
                #if KMP_TASK_AFFINITY_MEASURE_TIME
                    double time4 = get_wall_time2();
                #endif
                  tmp_result.data_domain = current_data_domain;
                  tmp_result.gtid = target_gtid;
                  KA_TRACE(5, ("__kmpc_omp_task: T#%d Setting initial mapping %lx ==> %d, %d\n", gtid, page_start_address, current_data_domain, target_gtid));
                  #if KMP_TASK_AFFINITY_USE_DEFAULT_MAP
                    __kmp_acquire_bootstrap_lock(&lock_addr_map);
                    task_aff_addr_map[page_start_address].data_domain = current_data_domain;
                    task_aff_addr_map[page_start_address].gtid = target_gtid;
                    __kmp_release_bootstrap_lock(&lock_addr_map);
                  #else
                    cur_entry->val.data_domain = current_data_domain;
                    cur_entry->val.gtid = target_gtid;
                  #endif
                  #if KMP_TASK_AFFINITY_MEASURE_TIME
                    time4 = get_wall_time2() - time4;
                    thread->th.th_sum_time_map_insert += time4;
                    thread->th.th_num_map_insert++;
                  #endif
                return tmp_result;
            }
        }
    }
      task_aff_physical_data_location_t tmp_result;
      tmp_result.data_domain = -2;
      tmp_result.gtid = -2;
      return tmp_result;
}

/* 
//searches page_loc[naffine][row] array for the dependency to use (weighted), based on the strategy to use
void inline map_count_weighted(kmp_task_affinity_info *aff_info, const int naffin,const int row, int page_loc[naffin][row], int array_size[naffin], int* x, int* y, int strat){
    const int page_size = KMP_GET_PAGE_SIZE();
    KA_TRACE(50,("+++ start count_weight strat w%d pageloc[0][0} %d naffin %d row %d array_size[0] %d\n",strat,page_loc[0][0], naffin, row, array_size[0]));

    if (strat == 1){
        //overall, no further balancing
        int max = 0;
        int* cur;
        std::map<int,int> m;
        for (int i=0; i < naffin; i++) {
            for (int j=0; j < array_size[i]; j++){
                cur = &page_loc[i][j];
                if (*cur < 0){continue;}
                m[*cur]++;
                if (m[*cur] > max) {
                    max = m[*cur];
                    *x=i;
                    *y=j;
                }
            }
        }
        return;
    }

    else if (strat == 2){
        //balace by affinity, every affinity as important, if row not full weight ea entry a bit more
        int max = 0;
        int* cur;
        std::map<int,double> m;
        for (int i=0; i < naffin; i++) {
            double weight = 1 + ( (row-array_size[i]) / (double) array_size[i]);//weight each entry as if every row is full
            KA_TRACE(50,("++weight %f row %d size %d\n",weight, row, array_size[i]));
            for (int j=0; j < array_size[i]; j++){
                cur = &page_loc[i][j];
                if (*cur < 0){continue;}
                m[*cur]+=weight;
                if (m[*cur] > max) {
                    max = m[*cur];
                    *x=i;
                    *y=j;
                }
            }
        }
        return;
    }

    else if (strat == 3){
        //balance by size, evenly distr array weight
        int max = 0;
        int* cur;
        std::map<int,double> m;
        int f = page_size*row; //divide weight by factor, to prevent overflow
        for (int i=0; i < naffin; i++) {
            double weight = 1 + ( (row-array_size[i]) / (double) array_size[i]);//weight each entry as if every row is full
            for (int j=0; j < array_size[i]; j++){
                cur = &page_loc[i][j];
                if (*cur < 0){continue;}
                m[*cur]+=weight*f;
                if (m[*cur] > max) {
                    max = m[*cur];
                    *x=i;
                    *y=j;
                }
            }
        }
        KA_TRACE(50,("++size max %d",max));
        return;
    }

    else{ //if strat == 0
        //choose first affinity, first entry
        *x=0;
        *y=0;
        return;
    }
}
*/
inline int get_cur(task_aff_physical_data_location_t page_loc)
{
  switch (kmp_affinity_settings.affinity_map_mode)
  {
    case kmp_affinity_map_type_domain:
      return page_loc.data_domain;
      break;
    case kmp_affinity_map_type_combined:
      return page_loc.data_domain;
      break;
    case kmp_affinity_map_type_thread:
      return page_loc.gtid;
      break;
  }
}

inline task_aff_physical_data_location_t map_count_weighted(kmp_task_affinity_info *aff_info, const int naffin,const int row, task_aff_physical_data_location_t page_loc[naffin][row], int array_size[naffin]) 
{
  const int page_size = KMP_GET_PAGE_SIZE();
    int x = 0, y = 0;
    int max = 0;
    int cur;
    std::map<int,int> m;

  switch (kmp_affinity_settings.page_weighting_strategy)
  {
  case kmp_affinity_page_weight_mode_first_page_only:
    return page_loc[0][0];
    break;
  case kmp_affinity_page_weight_mode_majority:
    x = 0;
    y = 0;
    max = 0;
    for (int i=0; i < naffin; i++) {
        for (int j=0; j < array_size[i]; j++){
            cur = get_cur(page_loc[i][j]);
            //
            if (cur < 0) {
              continue;
            }
            m[cur]++;
            if (m[cur] > max) {
                max = m[cur];
                x=i;
                y=j;
            }
        }
    }
    break;
  case kmp_affinity_page_weight_mode_by_affinity:
    max = 0;
    for (int i=0; i < naffin; i++) {
        double weight = 1 + ( (row-array_size[i]) / (double) array_size[i]);//weight each entry as if every row is full
        KA_TRACE(50,("++weight %f row %d size %d\n",weight, row, array_size[i]));
        for (int j=0; j < array_size[i]; j++){
            cur = get_cur(page_loc[i][j]);
            if (cur < 0){
              continue;
            }
            m[cur]+=weight;
            if (m[cur] > max) {
                max = m[cur];
                x=i;
                y=j;
            }
        }
    }
    break;
  case kmp_affinity_page_weight_mode_by_size:
    max = 0;
    int f = page_size*row; //divide weight by factor, to prevent overflow
    for (int i=0; i < naffin; i++) {
        double weight = 1 + ( (row-array_size[i]) / (double) array_size[i]);//weight each entry as if every row is full
        for (int j=0; j < array_size[i]; j++){
            cur = get_cur(page_loc[i][j]);
            if (cur < 0) {
              continue;
            }
            m[cur]+=weight*f;
            if (m[cur] > max) {
                max = m[cur];
                x=i;
                y=j;
            }
        }
    }
    break;
  }
  
  return page_loc[x][y];
}

task_aff_physical_data_location_t inline affinity_schedule(void **pointer2, kmp_int32 gtid, kmp_info_t *thread, kmp_int32 naffin, kmp_task_affinity_info *aff_info){
    #if KMP_TASK_AFFINITY_MEASURE_TIME
        double time1 = get_wall_time2();
    #endif



    void *pointer = *pointer2;
    const int page_size = KMP_GET_PAGE_SIZE();
    KA_TRACE(20, ("affinity_schedule (enter): T#%d #registred affinities %d\n",gtid, naffin));
    KA_TRACE(50, ("+++ aff_data[0,len] addr: %p %p length %d, page_size %d n %d\n", aff_info[0].base_addr, aff_info[0].base_addr+aff_info[0].len, aff_info[0].len, page_size, kmp_affinity_settings.number_of_affinities));
    KA_TRACE(50,("+++ domain of data 0 in map: %d\n",task_aff_addr_map.find(aff_info[0].base_addr & ~(page_size-1))->second))
    kmp_task_team_t *task_team = thread->th.th_task_team;
    kmp_thread_data_t *threads_data = (kmp_thread_data_t *)TCR_PTR(task_team->tt.tt_threads_data);
    size_t page_start_address;
    //int current_data_domain = -1;
    //kmp_info_t * target_thread = nullptr;

    if (kmp_affinity_settings.number_of_affinities < 1)
    {
      kmp_affinity_settings.number_of_affinities = 1;
    }
    const int n = kmp_affinity_settings.number_of_affinities;//for strat

    int max_len = n;//for loc array

    /*----------------------------------------------------------------*/
    /*determine the max page length dependent of the chosen strategy*/
    /*----------------------------------------------------------------*/

    if(kmp_affinity_settings.page_weighting_strategy == 0) {
      max_len = 1;
    } else {
      switch (kmp_affinity_settings.page_selection_strategy)
      {
        case kmp_affinity_page_mode_first_page_of_first_affinity_only:
          max_len = 1;
          break;
        case kmp_affinity_page_mode_divide_in_n_pages:
          max_len = n;
          break;
        case kmp_affinity_page_mode_every_nth_page:
          max_len = 1;
          for (int i=0; i < naffin; i++){
            if (aff_info[i].len > max_len){
                max_len = aff_info[i].len;
            }
          }
          max_len = ((max_len/page_size)/n)+1;
          break;
        case kmp_affinity_page_mode_first_and_last_page:
          max_len = 2;
          break;
        case kmp_affinity_page_mode_continuous_binary_search:
          max_len = n;
          break;
        case kmp_affinity_page_mode_first_page:
          max_len = 1;
          break;
      }
    }

    const int row = max_len;

    task_aff_physical_data_location_t page_loc[naffin][row];
    int array_size[naffin];//filled page loc array size
    int skipLen[naffin];

    for (int i = 0; i < naffin; i++) {
      array_size[i] = 0;
      skipLen[i] = 0;
      for (int j = 0; j < row; j++) {
          page_loc[i][j].gtid = -1;
          page_loc[i][j].data_domain = -1;
      }
    }

    int skip=0;

    switch (kmp_affinity_settings.page_selection_strategy)
    {
      case kmp_affinity_page_mode_first_page_of_first_affinity_only:

        page_loc[0][0] = check_page(gtid, thread, aff_info[0].base_addr);
        array_size[0] = 1;

        break;
      case kmp_affinity_page_mode_divide_in_n_pages:

        for (int i=0;i<naffin;i++){
            skipLen[i] = aff_info[i].len/n;
            array_size[i] = row;
            skip = 0;
            if (skipLen[i] < page_size){
                //e.g. for arrays with 1 element or high n
                //insted check every page contained in len
                skipLen[i] = page_size;
                array_size[i] = ((aff_info[i].len-1)/page_size)+1; //round up
            }
            for (int j=0; j < array_size[i]; j++){
                page_loc[i][j] = check_page(gtid, thread, aff_info[i].base_addr + skip);
                skip += skipLen[i];
            }
        }

        break;
      case kmp_affinity_page_mode_every_nth_page:

        for (int i=0; i < naffin; i++){
            skipLen[i] = page_size*n;
            skip = 0;
            array_size[i] = row;
            //KA_TRACE(50,("strat s2, skip %d size %d row %d\n", skipLen[i],array_size[i], row));
            for (int j=0; j < row; j++){
                page_loc[i][j] = check_page(gtid, thread, aff_info[i].base_addr + skip);

                skip += skipLen[i];
                if (skip >= aff_info[i].len) {
                    array_size[i]=j+1;
                }
            }
        }

        break;
      case kmp_affinity_page_mode_first_and_last_page:
        
        for (int i=0; i < naffin; i++) {
            array_size[i]=1;
            skipLen[i] = aff_info[i].len-1;
            //KA_TRACE(50,("strat s3, len %d, skipLen %d\n",aff_info[i].len, skipLen[i]));
            page_loc[i][0] = check_page(gtid, thread, aff_info[i].base_addr);

            if (skipLen[i] >= page_size) {
                //only check last page, if not on same page
                array_size[i]=2;
                page_loc[i][1] = check_page(gtid, thread, aff_info[i].base_addr + skipLen[i]);
            }
        }

        break;
      case kmp_affinity_page_mode_continuous_binary_search:

        KA_TRACE(50,("strat s4\n"));
        for (int i=0; i < naffin; i++){
            array_size[i] = n;
            skipLen[i] = aff_info[i].len/n;//for page_boundary_addr
            task_aff_physical_data_location_t top, bot, m;
            int half = aff_info[i].len/2;
            top = check_page(gtid, thread, aff_info[i].base_addr + aff_info[i].len-1);
            bot = check_page(gtid, thread, aff_info[i].base_addr);
            while (half >= page_size && top != bot){
                m = check_page(gtid, thread, aff_info[i].base_addr + half);
                half = half/2;
                if (m == bot){
                    bot = m;
                } else{ //m == top
                    top = m;
                }
            }
            //fill page_loc array up by same % based on found cut
            int scale = (half*row)/aff_info[i].len;
            for (int j=0; j < row; j++){
                if (j <= scale)
                  page_loc[i][j]=bot;
                else
                  page_loc[i][j]=top;
            }
        }

        break;
      case kmp_affinity_page_mode_first_page:
        for (int i=0; i < naffin; i++){
            page_loc[i][0] = check_page(gtid, thread, aff_info[i].base_addr);
            array_size[i] = 1;
        }
        break;
    }

    #if KMP_TASK_AFFINITY_MEASURE_TIME
        time1 = get_wall_time2() - time1;
        thread->th.th_sum_time_strategy1 += time1;
        thread->th.th_num_strategy1++;
        time1 = get_wall_time2();
    #endif
    /* 
    int x=0;
    int y=0;
    //count most common loc in page_loc with weight
    map_count_weighted(aff_info, naffin, row, page_loc, array_size, &x,&y, page_weighting_strategy);
    pointer = (void *) ((aff_info[x].base_addr + ((y*skipLen[y]) %aff_info[x].len) ) & ~(page_size-1));
    KA_TRACE(1, (" affinity_schedule (exit): loc %d (%d %d), pointer %p\n", page_loc[x][y],x,y,pointer));
    
    return page_loc[x][y];*/
    #if KMP_TASK_AFFINITY_MEASURE_TIME
        time1 = get_wall_time2() - time1;
        thread->th.th_sum_time_strategy2 += time1;
        thread->th.th_num_strategy2++;
    #endif

    return map_count_weighted(aff_info, naffin, row, page_loc, array_size);
}

kmp_int32 __kmpc_omp_task(ident_t *loc_ref, kmp_int32 gtid,
                          kmp_task_t *new_task) {
  kmp_int32 res;
  KMP_SET_THREAD_STATE_BLOCK(EXPLICIT_TASK);

#if KMP_DEBUG || OMPT_SUPPORT || KMP_USE_TASK_AFFINITY
  kmp_taskdata_t *new_taskdata = KMP_TASK_TO_TASKDATA(new_task);
#endif
  kmp_info_t * tmpthread = __kmp_threads[gtid];
  KA_TRACE(10, ("__kmpc_omp_task(enter): T#%d loc=%p task=%p current_task:%p parent_of_current:%p\n", gtid, loc_ref,
                new_taskdata, tmpthread->th.th_current_task, tmpthread->th.th_current_task->td_parent));

#if KMP_TASK_AFFINITY_MEASURE_TIME
  double t_overall = get_wall_time2();
  double time1, time2;
#endif

#if OMPT_SUPPORT
  kmp_taskdata_t *parent = NULL;
  if (__builtin_expect(ompt_enabled.enabled && !new_taskdata->td_flags.started,0)) {
    parent = new_taskdata->td_parent;
    if (!parent->ompt_task_info.frame.reenter_runtime_frame)
      parent->ompt_task_info.frame.reenter_runtime_frame =
          OMPT_GET_FRAME_ADDRESS(1);
    if (ompt_enabled.ompt_callback_task_create) {
      ompt_data_t task_data = ompt_data_none;
      ompt_callbacks.ompt_callback(ompt_callback_task_create)(
          parent ? &(parent->ompt_task_info.task_data) : &task_data,
          parent ? &(parent->ompt_task_info.frame) : NULL,
          &(new_taskdata->ompt_task_info.task_data),
          ompt_task_explicit | TASK_TYPE_DETAILS_FORMAT(new_taskdata), 0,
          OMPT_GET_RETURN_ADDRESS(0));
    }
  }
#endif

#if KMP_USE_TASK_AFFINITY
  res = __kmpc_omp_task_affinity(loc_ref, gtid, new_task);
#else
  res = __kmp_omp_task(gtid, new_task, true);
#endif

  KA_TRACE(10, ("__kmpc_omp_task(exit): T#%d returning "
                "TASK_CURRENT_NOT_QUEUED: loc=%p task=%p\n",
                gtid, loc_ref, new_taskdata));
#if OMPT_SUPPORT
  if (__builtin_expect(ompt_enabled.enabled && parent != NULL,0)) {
    parent->ompt_task_info.frame.reenter_runtime_frame = NULL;
  }
#endif
#if KMP_TASK_AFFINITY_MEASURE_TIME
  t_overall = get_wall_time2() - t_overall;
  tmpthread->th.th_sum_time_kmpc_omp_task += t_overall;
#endif
  return res;
}

#if KMP_USE_TASK_AFFINITY
kmp_int32 __kmpc_omp_task_affinity(ident_t *loc_ref, kmp_int32 gtid, kmp_task_t *new_task)
{
#if KMP_TASK_AFFINITY_MEASURE_TIME
  double t_overall = get_wall_time2();
  double time1, time2, time3;
  time3 = get_wall_time2();
#endif
  kmp_int32 res;
  
  kmp_info_t *thread = __kmp_threads[gtid];
  kmp_taskdata_t *new_taskdata = KMP_TASK_TO_TASKDATA(new_task);
  thread->th.th_count_overall_tasks_generated++;

  if(thread->th.th_task_affinity_msg)
  {
    // print debug msg that has been passed by program
    fprintf(stderr, "th_task_affinity_msg: T#%d task=%p current_task:%p parent_task:%p Message:%s\n", gtid, new_taskdata, thread->th.th_current_task, new_taskdata->td_parent, thread->th.th_task_affinity_msg);
    thread->th.th_task_affinity_msg = NULL;
  }

  KA_TRACE(50,("+++++ tasking_mode %d\n", __kmp_tasking_mode));
  if(__kmp_tasking_mode == tskm_immediate_exec)
  {
      KA_TRACE(50,("Immediate exec!!\n"))
    // do not use task queue but execute immediately
    res = __kmp_omp_task(gtid, new_task, true);
  } else {
    kmp_task_team_t *task_team = thread->th.th_task_team;

    // if single threaded
    if(task_team == NULL)
    {
      res = __kmp_omp_task(gtid, new_task, true);
    } else {
      kmp_int32 nthreads_in_team = task_team->tt.tt_nproc;

      // need to enable tasking and allocate and assign proper data structures once
      if (!KMP_TASKING_ENABLED(task_team)) {
        __kmp_acquire_bootstrap_lock(&lock_enable_task_team);
        if (!KMP_TASKING_ENABLED(task_team)) {
          __kmp_enable_tasking(task_team, thread);
        }
        __kmp_release_bootstrap_lock(&lock_enable_task_team);
      }

      kmp_thread_data_t *threads_data = (kmp_thread_data_t *)TCR_PTR(task_team->tt.tt_threads_data);
      KMP_DEBUG_ASSERT(__kmp_tasking_mode != 0);

      //__kmpc_omp_reg_thread_with_affinity(gtid, new_task);

      //if(thread->th.naffin == 0 || nthreads_in_team <= 1)
      if(new_taskdata->naffin == 0 || nthreads_in_team <= 1)
      {
#if KMP_TASK_AFFINITY_MEASURE_TIME
        time1 = get_wall_time2();
#endif
        // KA_TRACE(5, ("TASK AFFINITY: __kmpc_omp_task: T#%d task_affinity_data is NULL.\n", gtid));
        res = __kmp_omp_task(gtid, new_task, true);
#if KMP_TASK_AFFINITY_MEASURE_TIME
        time1 = get_wall_time2()-time1;
        thread->th.th_sum_time_pushing += time1;
        thread->th.th_num_pushing++;
#endif
      }
      else
      {
#if KMP_TASK_AFFINITY_MEASURE_TIME
        time1 = get_wall_time2();
#endif
        // Allocate deque if necessary
        kmp_int32 tid = __kmp_tid_from_gtid(gtid);
        kmp_thread_data_t *cur_thread_data = &(threads_data[tid]);

        int current_data_domain = -1;
        int ret_code = -1;
        int target_tid = -1;
        int target_gtid = -1;
        kmp_info_t * target_thread = NULL;
        void *page_boundary_pointer;

        //task_aff_physical_data_location_t loc = affinity_schedule(&page_boundary_pointer, gtid, thread, thread->th.naffin, thread->th.th_task_affinity_data);
        task_aff_physical_data_location_t loc = affinity_schedule(&page_boundary_pointer, gtid, thread, new_taskdata->naffin, new_taskdata->affinity_info);

        current_data_domain = loc.data_domain;
        target_gtid = loc.gtid;

        switch (kmp_affinity_settings.affinity_map_mode) {
          case kmp_affinity_map_type_combined:

              KA_TRACE(3, ("_kmpc_omp_task_affinity: T#%d compare domain from Thread T#%d in numa domain %d with Target numa domain: %d\n", gtid, target_gtid,map_thread_to_numa_domain[target_gtid], current_data_domain));
              
              if(target_gtid >= 0 && current_data_domain >= 0)
              {
                thread->th.th_combined_strat_scheduled_overall++;
                target_tid = __kmp_tid_from_gtid(target_gtid);
                kmp_thread_data_t *target_threads_data;
                target_threads_data = &threads_data[target_tid];
                int num_tasks = target_threads_data->td.td_deque_ntasks;

                if(current_data_domain == map_thread_to_numa_domain[target_gtid] && num_tasks < 256)
                {
                  KA_TRACE(3, ("_kmpc_omp_task_affinity: Found thread in combined map where task_data from same numa domain was previously used\n", gtid));
                  //suche_hier
                  bool found = false;
                  KA_TRACE(3, ("curr data domain: %d\n", current_data_domain ));
                  int numa_domain_size = -1;
                  if (task_team->tt.tt_numa_domains_set) 
                    numa_domain_size = task_team->tt.tt_numa_domain_size[current_data_domain];
                  KA_TRACE(2, ("numa_domain_size: %d\n", numa_domain_size));
                  double threshold = kmp_affinity_settings.threshold_for_thread_selection;
                  int tmp_tid = -1;
                  kmp_thread_data_t *tmp_threads_data;
                  int tmp_num_tasks = 256;
                  kmp_int32 thread_lowest = -1;
                  kmp_int32 thread_lowest_num_tasks = 256;


                  if(num_tasks == 0) 
                  {
                    found = true;
                    KA_TRACE(3, ("_kmpc_omp_task_affinity: queue is (nearly) emtpy, push task to queue!\n"));
                  } else {
                    if (numa_domain_size >= 0){
                      //found = true;

                      for (int i = 0; i < numa_domain_size; i++)
                      { 
                        int tmp_gtid = task_team->tt.tt_map_threads_in_domain[current_data_domain][i];
                        
                        tmp_tid = __kmp_tid_from_gtid(tmp_gtid);
                        tmp_threads_data = &threads_data[tmp_tid];
                        tmp_num_tasks = tmp_threads_data->td.td_deque_ntasks;
                        
                        if(tmp_num_tasks == 0) 
                        {
                          target_gtid = tmp_gtid;
                          found = true;
                          thread->th.th_combined_strat_found_other_empty_thread++;
                          break;
                        }

                        KA_TRACE(3, ("min: %d and thread %d:\n", tmp_num_tasks, tmp_gtid));
                        if(tmp_num_tasks < thread_lowest_num_tasks)
                        {
                          thread_lowest_num_tasks = tmp_num_tasks;
                          thread_lowest = tmp_gtid;
                        }
                      } 

                      if(!found)
                      {
                        double task_ratio = ((double)(thread_lowest_num_tasks))/((double)(num_tasks));

                        KA_TRACE(3,("num task is zero!, error: %d, %d\n", num_tasks, thread_lowest_num_tasks));
                        KA_TRACE(3,("ratio %lf\n",task_ratio));
                        KA_TRACE(3,("threshold: %lf\n", threshold))

                        
                        if (task_ratio <= threshold)
                        {
                          found = true;
                          thread->th.th_combined_strat_found_other_thread_under_threshold++;
                        } else {
                          thread->th.th_combined_strat_pushed_over_treshold++;

                          found = true;
                          target_gtid = thread_lowest;
                        }
                      }
                    }
                  }

                  if(found)
                  {
                    target_tid = __kmp_tid_from_gtid(target_gtid);
                    new_taskdata->td_task_affinity_data_domain = current_data_domain;
                    break;
                  }
                }
              }
          case kmp_affinity_map_type_domain:
            target_gtid = -1;
            if(current_data_domain < 0) {
              break;
            }

            if(kmp_affinity_settings.thread_selection_strategy == kmp_affinity_thread_selection_mode_private
                    && current_data_domain == thread->th.th_task_aff_my_domain_nr) {
                target_gtid = gtid;
                target_tid = __kmp_tid_from_gtid(target_gtid);
            } else {
                KA_TRACE(50,("calling init thread: c2 domain %d, team %d, threads %d, tid %d, gtid %d, idx %d\n",current_data_domain, task_team, threads_data, tid, gtid, threads_data->td.td_idx_in_numa_map));
                target_thread = __kmp_task_aff_get_initial_thread_in_numa_domain(current_data_domain, task_team, threads_data, &target_tid, &target_gtid);
            }
            new_taskdata->td_task_affinity_data_domain = current_data_domain;
            break;
          case kmp_affinity_map_type_thread:
            current_data_domain = -1;
            if(target_gtid < 0)
              break;

            #if KMP_TASK_AFFINITY_MEASURE_TIME
                time2 = get_wall_time2();
            #endif

            int tmp_err = move_pages(0 /*self memory */, 1, &page_boundary_pointer, NULL, &current_data_domain, 0);
            #if KMP_TASK_AFFINITY_MEASURE_TIME
              time2 = get_wall_time2()-time2;
              thread->th.th_sum_time_identify_physical_location += time2;
              thread->th.th_num_identify_physical_location++;
            #endif
            if(tmp_err == 0 && current_data_domain >= 0)
              new_taskdata->td_task_affinity_data_domain = current_data_domain;
              // DEBUG
              // set data domain to some value >= 0; not really used but necessary to not run into fallback mode
              current_data_domain = 0;
              target_tid = __kmp_tid_from_gtid(target_gtid);
            break;
        }

        // reset pointer & save temporary
        //__kmpc_omp_reg_task_with_affinity(loc_ref, gtid, new_task, thread->th.naffin, thread->th.th_task_affinity_data);


        //new_taskdata->affinity_info = thread->th.th_task_affinity_data;
        //new_taskdata->naffin = thread->th.naffin;
        if (thread->th.naffin > 0){
          thread->th.naffin = 0;
  #if KMP_TASK_AFFINITY_NEW_MEMORY_ALLOC
          free(thread->th.th_task_affinity_data);
          //thread->th.th_task_affinity_data = nullptr;
  #else
          thread->th.th_task_affinity_data = (kmp_task_affinity_info_t*) malloc(4 * sizeof(kmp_task_affinity_info_t));
  #endif         
        }

#if KMP_TASK_AFFINITY_MEASURE_TIME
        time1 = get_wall_time2()-time1;
        thread->th.th_sum_time_map_overall += time1;
        thread->th.th_num_map_overall++;

        time1 = get_wall_time2();
#endif

        KA_TRACE(3, ("_kmpc_omp_task_affinity: Strategy took Thread T#%d in numa domain %d with Target numa domain: %d\n", target_gtid,map_thread_to_numa_domain[target_gtid], current_data_domain));

        if(current_data_domain >= 0) {
          if(target_tid == -1)
          {
            // fall back mode if not possible to find any matching thread
            KA_TRACE(5, ("__kmpc_omp_task: T#%d fallback mode 1 (should not happen under normal circumstances) , target_id: %d\n", gtid, target_tid));
            thread->th.th_count_task_pushed_in_fallback_mode1++;
            res = __kmp_omp_task(gtid, new_task, true);
          } else {
            if (gtid == target_gtid) {
              res = __kmp_omp_task(gtid, new_task, true);
            } else {
#if KMP_TASK_AFFINITY_MEASURE_TIME
              time2 = get_wall_time2();
#endif
              res = __kmp_omp_task_aff(gtid, target_gtid, new_task, true);
#if KMP_TASK_AFFINITY_MEASURE_TIME
              time2 = get_wall_time2()-time2;
              thread->th.th_sum_time_pushing_inaff += time2;
              thread->th.th_num_pushing_inaff++;
#endif
            }
          }
        } else {
          // fall back mode if not possible to find any matching thread or not a correct numa domain for poninter
          KA_TRACE(5, ("__kmpc_omp_task: T#%d fallback mode, current_data_domain %d\n", gtid, current_data_domain));
          thread->th.th_count_task_pushed_in_fallback_mode2++;
          res = __kmp_omp_task(gtid, new_task, true);
        } // if
#if KMP_TASK_AFFINITY_MEASURE_TIME
        time1 = get_wall_time2()-time1;
        thread->th.th_sum_time_pushing += time1;
        thread->th.th_num_pushing++;
#endif
      }
    }
  }

#if KMP_TASK_AFFINITY_MEASURE_TIME
  t_overall = get_wall_time2() - t_overall;
  thread->th.th_sum_time_kmpc_omp_task += t_overall;
#endif


#if KMP_TASK_AFFINITY_MEASURE_TIME
  time1 = get_wall_time2();
  time3 = get_wall_time2()-time3;
  new_taskdata->td_ts_strat_execution += time3;
#endif

  return res;
}
#endif

#if KMP_USE_TASK_AFFINITY
void __kmpc_set_task_affinity(void * data_start, int len)
{
    int gtid = __kmp_entry_gtid();
    kmp_info_t *thread = __kmp_threads[gtid];
    KA_TRACE(30, ("__kmpc_set_task_affinity(enter): T#%d &data_start %p, len %d, number %d\n",
        gtid, (kmp_intptr_t) data_start, len, thread->th.naffin));
    //new info struct with data
    kmp_task_affinity_info_t task_affinity_info;
    task_affinity_info.base_addr= (kmp_intptr_t) data_start;
    if (len < 1){len = 1;}//len <1 set to 1
    task_affinity_info.len= len;
    #if KMP_TASK_AFFINITY_MEASURE_TIME
        if (len > thread->th.th_count_max_aff_data_len)
        thread->th.th_count_max_aff_data_len = len;
    #endif
    KA_TRACE(50, ("__kmpc_set_task_affinity: T#%d size %d, affinities %d, INIT type (weight) %d (strategy) %d and num %d, len %d\n",
                gtid, sizeof(kmp_task_affinity_info_t), thread->th.naffin, kmp_affinity_settings.page_weighting_strategy, kmp_affinity_settings.page_selection_strategy , kmp_affinity_settings.number_of_affinities, task_affinity_info.len));

    //malloc for 4, realloc ea time > 4.
  #if KMP_TASK_AFFINITY_NEW_MEMORY_ALLOC
    if (thread->th.naffin == 0) {
        thread->th.th_task_affinity_data = (kmp_task_affinity_info_t*) malloc(4 * sizeof(kmp_task_affinity_info_t));
    } else
  #endif
    if (thread->th.naffin>=4){
        thread->th.th_task_affinity_data = (kmp_task_affinity_info_t*) realloc(thread->th.th_task_affinity_data, (thread->th.naffin + 1) * sizeof(kmp_task_affinity_info_t));
    }
        thread->th.th_task_affinity_data[thread->th.naffin] = task_affinity_info;
        thread->th.naffin++;
}

void __kmpc_task_affinity_taskexectimes_set_enabled( int enabled )
{
    // just set global parameter
    taskexectimes_enabled = enabled;
}

//aff_schedule is composed number of schedule to use
//aff_num is parameter for schedule (e.g. divide in n parts)
void __kmpc_task_affinity_init(kmp_affinity_settings_t affinity_settings)
{
  kmp_affinity_settings = affinity_settings;

  enable_numa_aware_stealing = true;
}

void __kmpc_task_affinity_set_msg(char * msg) {
  kmp_info_t *thread = __kmp_threads[__kmp_entry_gtid()];
  thread->th.th_task_affinity_msg = msg;
}

inline bool __kmp_task_aff_is_correct_task(
  kmp_info_t * thread,
  kmp_taskdata_t *taskdata,
  kmp_thread_data_t *victim_td,
  kmp_task_team_t * task_team) {

  // check against all current tasks of threads in team due to adding task to queues of different threads
  int min_lvl = 1;
  bool min_set = false;
  kmp_taskdata_t *parent = NULL;
  int gtid = __kmp_entry_gtid();
  kmp_taskdata_t *current_task;

  parent = taskdata->td_parent;

  // first check current thread
  int tmp_tid = __kmp_tid_from_gtid(gtid);
  current_task = thread->th.th_current_task;
  kmp_int32 cur_lvl = current_task->td_level;

  while (parent != current_task && parent->td_level > cur_lvl) {
    parent = parent->td_parent; // check generation up to the level of the
    // KA_TRACE(10, ("TASK AFFINITY: T#%d Check parent level of thread %d ntasks=%d head=%u tail=%u -- curtask:%p curlvl:%d ptask:%p plvl:%d\n",
    //               gtid, tmp_tid, victim_td->td.td_deque_ntasks,
    //               victim_td->td.td_deque_head, victim_td->td.td_deque_tail, current_task, cur_lvl, parent, parent->td_level));
    // current task
    // KMP_DEBUG_ASSERT(parent != NULL);
  }
  if(current_task == parent) {
    return true;
  }

  // reset pointer
  parent = taskdata->td_parent;
  int tmp_nproc = task_team->tt.tt_nproc;
  // kmp_taskdata_t * current_task;

  while(parent->td_level >= min_lvl)
  {
    for(int i = 0; i < tmp_nproc; i++) {
      current_task = __kmp_threads[__kmp_gtid_from_tid(i, thread->th.th_team)]->th.th_current_task;
      // KA_TRACE(10, ("TASK AFFINITY: T#%d Check parent level of thread %d ntasks=%d head=%u tail=%u -- curtask:%p curlvl:%d ptask:%p plvl:%d\n",
      //               gtid, i, victim_td->td.td_deque_ntasks,
      //               victim_td->td.td_deque_head, victim_td->td.td_deque_tail, current_task, cur_lvl, parent, parent->td_level));
      if(current_task == parent)
        return true;

      if(!min_set) {
        kmp_int32 cur_lvl = current_task->td_level;
        if(cur_lvl < min_lvl)
          min_lvl = cur_lvl;
      }
    }
    // ran once through, so should be fine here
    min_set = true;
    // go to next parent
    parent = parent->td_parent;
    if(parent == NULL)
      return false;
  }
  return false;
}

inline kmp_info_t * __kmp_task_aff_get_initial_thread_in_numa_domain (
  int current_data_domain,
  kmp_task_team_t *task_team,
  kmp_thread_data_t *threads_data,
  int* target_tid,
  int* target_gtid) {

  // KA_TRACE(10, ("__kmp_task_aff_get_initial_thread_in_numa_domain: T#%d Initial thread type is %d\n",
  //                 __kmp_entry_gtid(), task_aff_init_thread_type));

  // get threads for numa domain & determine tid and gtid where task should be scheduled
  kmp_info_t * target_thread = NULL;
  kmp_thread_data_t *target_thread_data = NULL;
  kmp_thread_data_t *my_thread_data = NULL;
  *target_tid = -1;
  *target_gtid = -1;

  // if not all threads registered in global list
  if(!numa_all_set_up) //TODO: RACE?
    return NULL;

  int nthreads = task_team->tt.tt_nproc;

  // set numa mapping for current task team once
  if(!task_team->tt.tt_numa_domains_set)
  {
    __kmp_acquire_bootstrap_lock( &task_team->tt.tt_lock_numa_map );
    if(!task_team->tt.tt_numa_domains_set)
    {
      double t1;
      int i;
#if KMP_TASK_AFFINITY_MEASURE_TIME
      t1 = get_wall_time2();
#endif
      KA_TRACE(10, ("__kmp_task_aff_get_initial_thread_in_numa_domain: T#%d Setting up numa map for task team.\n", __kmp_entry_gtid()));
      // get max number of domains to init
      for (i = NUMA_DOMAIN_MAX_NR-1; i >= 0; i--)
      {
        if(numa_domain_size[i] != 0)
        {
          // set max number
          task_team->tt.tt_num_numa_domains = i+1;
          break;
        }
      }

      // allocate stuff
      task_team->tt.tt_numa_domain_size = (int*) malloc(task_team->tt.tt_num_numa_domains * sizeof(int));
      task_team->tt.tt_numa_domain_rr_counter = (int*) malloc(task_team->tt.tt_num_numa_domains * sizeof(int));
      //task_team->tt.tt_numa_domain_rr_locks = (kmp_bootstrap_lock_t*) malloc(task_team->tt.tt_num_numa_domains * sizeof(kmp_bootstrap_lock_t));

      task_team->tt.tt_map_threads_in_domain = (int**) malloc(task_team->tt.tt_num_numa_domains * sizeof(int *));
      for (i = 0; i < task_team->tt.tt_num_numa_domains; i++){
        task_team->tt.tt_numa_domain_size[i] = 0;
        task_team->tt.tt_numa_domain_rr_counter[i] = 0;
        task_team->tt.tt_map_threads_in_domain[i] = (int*) malloc(MAX_THREADS_PER_DOMAIN * sizeof(int));
      }

      for (i = 0; i < nthreads; i++) {
        // get tid & gid
        int cur_gtid = threads_data[i].td.td_thr->th.th_info.ds.ds_gtid;
        // get numa node
        int tmp_numa_domain = map_thread_to_numa_domain[cur_gtid];
        int tmp_count = task_team->tt.tt_numa_domain_size[tmp_numa_domain];
        task_team->tt.tt_map_threads_in_domain[tmp_numa_domain][tmp_count] = cur_gtid;
        task_team->tt.tt_numa_domain_size[tmp_numa_domain]++;
        KA_TRACE(10, ("setting domain[%d][%d] = %d.\n", tmp_numa_domain, tmp_count, cur_gtid));
      }
#if KMP_TASK_AFFINITY_MEASURE_TIME
      t1 = get_wall_time2()-t1;
#endif
      KA_TRACE(10, ("__kmp_task_aff_get_initial_thread_in_numa_domain: T#%d Time for setting up %f ms, nthreads=%d, tt_num_numa_domains=%d.\n", __kmp_entry_gtid(), t1, nthreads, task_team->tt.tt_num_numa_domains));

      for(i = 0; i < task_team->tt.tt_num_numa_domains; i++)
      {
        int tmp_size = task_team->tt.tt_numa_domain_size[i];
        for (int j = 0; j < tmp_size; j++)
        {
          KA_TRACE(10, ("domain[%d][%d] = %d.\n", i, j, task_team->tt.tt_map_threads_in_domain[i][j]));
        }
      }
      task_team->tt.tt_numa_domains_set = true;
      //fprintf(stderr, "__kmp_task_aff_get_initial_thread_in_numa_domain: T#%d Setting numa info for task team\n", __kmp_entry_gtid());
    }
    __kmp_release_bootstrap_lock( &task_team->tt.tt_lock_numa_map );
  }

  // if domain is not present in this team
  if(current_data_domain >= task_team->tt.tt_num_numa_domains)
    return NULL;
  // if domain is empty
  if(task_team->tt.tt_numa_domain_size[current_data_domain] == 0)
    return NULL;

  if(kmp_affinity_settings.thread_selection_strategy == kmp_affinity_thread_selection_mode_first) {
    *target_gtid = task_team->tt.tt_map_threads_in_domain[current_data_domain][0];
  } else if(kmp_affinity_settings.thread_selection_strategy == kmp_affinity_thread_selection_mode_lowest_wl || kmp_affinity_settings.thread_selection_strategy == kmp_affinity_thread_selection_mode_private) {
    int min_work = INT_MAX;
    int tmp_size = task_team->tt.tt_numa_domain_size[current_data_domain];
    int cur_id = -1;
    //__kmp_acquire_bootstrap_lock(&lock_domain_init_thread_region);
    for (int t = 0; t < tmp_size; t++)
    {
      // get thread
      int tmp_gtid = task_team->tt.tt_map_threads_in_domain[current_data_domain][t];
      int tmp_tid = __kmp_tid_from_gtid(tmp_gtid);
      target_thread_data = &threads_data[tmp_tid];

      // get current number of threads
      // __kmp_acquire_bootstrap_lock(&target_thread_data->td.td_deque_lock);
      int tmp_num_tasks = target_thread_data->td.td_deque_ntasks;
      // __kmp_release_bootstrap_lock(&target_thread_data->td.td_deque_lock);
      //double cur_time = get_wall_time2();
      //KA_TRACE(10, ("__kmp_task_aff_get_initial_thread_in_numa_domain:\tT#%d\t%f\ttarget thread with lowest tmp:\tT#%d\twith min work=\t%d.\n", __kmp_entry_gtid(), cur_time, tmp_gtid, tmp_num_tasks));

      if(tmp_num_tasks == 0){
        // immediatly quit here because no other thread can be lower
        cur_id = tmp_gtid;
        min_work = tmp_num_tasks;
        break;
      }
      if(tmp_num_tasks < min_work){
        cur_id = tmp_gtid;
        min_work = tmp_num_tasks;
      }
    }
    //__kmp_release_bootstrap_lock(&lock_domain_init_thread_region);
    *target_gtid = cur_id;

#if KMP_TASK_AFFINITY_MEASURE_TIME
    double cur_time = get_wall_time2();
#else
    double cur_time = -1;
#endif
    KA_TRACE(10, ("__kmp_task_aff_get_initial_thread_in_numa_domain:\tT#%d\t%f\ttarget thread with lowest number of tasks:\tT#%d\twith min work=\t%d\tdomain\t%d.\n", __kmp_entry_gtid(), cur_time, *target_gtid, min_work, current_data_domain));
  } else if(kmp_affinity_settings.thread_selection_strategy == kmp_affinity_thread_selection_mode_round_robin) {

    int cur_gtid = __kmp_entry_gtid();
    int cur_tid = __kmp_tid_from_gtid(cur_gtid);
    kmp_info_t * orig_thread = __kmp_threads[cur_gtid];
    //my_thread_data = &threads_data[cur_tid];

    // if(my_thread_data->td.td_idx_in_numa_map == -1)
    // {
    //   int tmp_size = task_team->tt.tt_numa_domain_size[orig_thread->th.th_task_aff_my_domain_nr];
    //   int i;
    //   for(i = 0; i < tmp_size; i++)
    //   {
    //     int tmp = task_team->tt.tt_map_threads_in_domain[orig_thread->th.th_task_aff_my_domain_nr][i];
    //     if(tmp == cur_gtid) {
    //       my_thread_data->td.td_idx_in_numa_map = i;
    //       break;
    //     }
    //   }
    // }

    if(orig_thread->th.th_numa_domain_rr_counter == NULL) {
      orig_thread->th.th_numa_domain_rr_counter = (int*) malloc(task_team->tt.tt_num_numa_domains * sizeof(int));
      for(int t = 0; t < task_team->tt.tt_num_numa_domains; t++)
        //orig_thread->th.th_numa_domain_rr_counter[t] = (my_thread_data->td.td_idx_in_numa_map) % task_team->tt.tt_numa_domain_size[t];
        orig_thread->th.th_numa_domain_rr_counter[t] = 0;
    }
    int tmp_counter = (orig_thread->th.th_numa_domain_rr_counter[current_data_domain]+1) % task_team->tt.tt_numa_domain_size[current_data_domain];
    orig_thread->th.th_numa_domain_rr_counter[current_data_domain] = tmp_counter;

    // __kmp_acquire_bootstrap_lock(&task_team->tt.tt_numa_domain_rr_locks[current_data_domain]);
    // int tmp_counter = (task_team->tt.tt_numa_domain_rr_counter[current_data_domain]+1) % task_team->tt.tt_numa_domain_size[current_data_domain];
    // task_team->tt.tt_numa_domain_rr_counter[current_data_domain] = tmp_counter;
    // __kmp_release_bootstrap_lock(&task_team->tt.tt_numa_domain_rr_locks[current_data_domain]);
    *target_gtid = task_team->tt.tt_map_threads_in_domain[current_data_domain][tmp_counter];
  } else {
    // // default random
    // srand(time(NULL));
    // int tmp_rand_idx = rand() % task_team->tt.tt_numa_domain_size[current_data_domain];
    kmp_info_t * orig_thread = __kmp_threads[__kmp_entry_gtid()];
    int tmp_rand_idx = __kmp_get_random(orig_thread) % task_team->tt.tt_numa_domain_size[current_data_domain];
    *target_gtid = task_team->tt.tt_map_threads_in_domain[current_data_domain][tmp_rand_idx];
  }

  *target_tid = __kmp_tid_from_gtid(*target_gtid);
  target_thread = __kmp_threads[*target_gtid];

  if(*target_tid == -1)
    return NULL;
  return target_thread;
}

void task_aff_print_numa_pinning(){
  //   for (int tmp = 0; tmp < 24; tmp++)
  //   {
  //     int cur_size = numa_domain_size[tmp];

  //     if( cur_size > 0 )
  //     {
  //       int pos_inner = 0;
  //       int tmp_nr_threads = 0;
  //       char string_per_map[256];

  //       for (int j = 0; j < cur_size; j++)
  //       {
  //         pos_inner += sprintf(&string_per_map[pos_inner], "%d ", map_threads_in_numa_domain[tmp][j]);
  //         tmp_nr_threads++;
  //       }
  //       KA_TRACE(10, ("TASK AFFINITY: NUMA DOMAIN %d has %d threads: %s\n", tmp, tmp_nr_threads, string_per_map));
  //     }
  //   }
}
#endif

/*!
@ingroup TASKING
@param loc_ref location of the original task directive
@param gtid Global Thread ID of encountering thread
@param new_task task thunk allocated by __kmpc_omp_task_alloc() for the ''new
task''
@param naffins Number of affinity items
@param affin_list List of affinity items
@return Returns non-zero if registering affinity information was not successful.
Returns 0 if registration was successful
This entry registers the affinity information attached to a task with the task thunk
structure kmp_taskdata_t.
*/
kmp_int32 __kmpc_omp_reg_task_with_affinity(ident_t *loc_ref, kmp_int32 gtid, kmp_task_t *new_task, kmp_int32 naffins, kmp_task_affinity_info_t *affin_list){
    kmp_taskdata_t *new_taskdata = KMP_TASK_TO_TASKDATA(new_task);
    //suche
    int num = 4;
    if (naffins >= 4)
      num = naffins + 1;
    
    new_taskdata->affinity_info = (kmp_task_affinity_info_t*) malloc(num * sizeof(kmp_task_affinity_info_t));

    for (int i = 0; i < naffins; i++)
    {
      new_taskdata->affinity_info[i] = affin_list[i];
    }
    new_taskdata->naffin = naffins;
    return 1;
}

kmp_int32 __kmpc_omp_reg_thread_with_affinity(kmp_int32 gtid, kmp_task_t *new_task)
{
    kmp_info_t *thread = __kmp_threads[gtid];
    kmp_taskdata_t *new_taskdata = KMP_TASK_TO_TASKDATA(new_task);
    
    int naffins = new_taskdata->naffin;
    int num = 4;
    if (naffins >= 4)
      num = naffins + 1;

    if (thread->th.naffin == 0 && naffins > 0)
    {
      thread->th.th_task_affinity_data = (kmp_task_affinity_info_t*) malloc(num * sizeof(kmp_task_affinity_info_t));

      for (int i = 0; i < naffins; i++)
      {
        thread->th.th_task_affinity_data[i] = new_taskdata->affinity_info[i];
      }

      thread->th.naffin = naffins;
    }

    return 1;
}

#if OMPT_SUPPORT
OMPT_NOINLINE
static kmp_int32 __ompt_enabled_taskwait(ident_t *loc_ref, kmp_int32 gtid,
    void *frame_address, void *return_address) {
  kmp_taskdata_t *taskdata;
  kmp_info_t *thread;
  int thread_finished = FALSE;
  KMP_SET_THREAD_STATE_BLOCK(TASKWAIT);

  KA_TRACE(10, ("__kmpc_omp_taskwait(enter): T#%d loc=%p\n", gtid, loc_ref));

  if (__kmp_tasking_mode != tskm_immediate_exec) {
    thread = __kmp_threads[gtid];
    taskdata = thread->th.th_current_task;

    ompt_data_t *my_task_data;
    ompt_data_t *my_parallel_data;

    my_task_data = &(taskdata->ompt_task_info.task_data);
    my_parallel_data = &(thread->th.th_team->t.ompt_team_info.parallel_data);

    taskdata->ompt_task_info.frame.reenter_runtime_frame = frame_address;

    if (ompt_enabled.ompt_callback_sync_region) {
      ompt_callbacks.ompt_callback(ompt_callback_sync_region)(
          ompt_sync_region_taskwait, ompt_scope_begin, my_parallel_data,
          my_task_data, return_address);
    }

    if (ompt_enabled.ompt_callback_sync_region_wait) {
      ompt_callbacks.ompt_callback(ompt_callback_sync_region_wait)(
          ompt_sync_region_taskwait, ompt_scope_begin, my_parallel_data,
          my_task_data, return_address);
    }
#if KMP_USE_TASK_AFFINITY && KMP_TASK_AFFINITY_MEASURE_TIME
  stop_task_execution_measurement(taskdata);
#endif

// Debugger: The taskwait is active. Store location and thread encountered the
// taskwait.
#if USE_ITT_BUILD
// Note: These values are used by ITT events as well.
#endif /* USE_ITT_BUILD */
    taskdata->td_taskwait_counter += 1;
    taskdata->td_taskwait_ident = loc_ref;
    taskdata->td_taskwait_thread = gtid + 1;

#if USE_ITT_BUILD
    void *itt_sync_obj = __kmp_itt_taskwait_object(gtid);
    if (itt_sync_obj != NULL)
      __kmp_itt_taskwait_starting(gtid, itt_sync_obj);
#endif /* USE_ITT_BUILD */

    bool must_wait =
        !taskdata->td_flags.team_serial && !taskdata->td_flags.final;

#if OMP_45_ENABLED
    must_wait = must_wait || (thread->th.th_task_team != NULL &&
                              thread->th.th_task_team->tt.tt_found_proxy_tasks);
#endif
    if (must_wait) {
      kmp_flag_32 flag(
          RCAST(volatile kmp_uint32 *, &taskdata->td_incomplete_child_tasks),
          0U);
      while (TCR_4(taskdata->td_incomplete_child_tasks) != 0) {
        flag.execute_tasks(thread, gtid, FALSE,
                           &thread_finished USE_ITT_BUILD_ARG(itt_sync_obj),
                           __kmp_task_stealing_constraint);
      }
    }
#if USE_ITT_BUILD
    if (itt_sync_obj != NULL)
      __kmp_itt_taskwait_finished(gtid, itt_sync_obj);
#endif /* USE_ITT_BUILD */

    // Debugger:  The taskwait is completed. Location remains, but thread is
    // negated.
    taskdata->td_taskwait_thread = -taskdata->td_taskwait_thread;

    if (ompt_enabled.ompt_callback_sync_region_wait) {
      ompt_callbacks.ompt_callback(ompt_callback_sync_region_wait)(
          ompt_sync_region_taskwait, ompt_scope_end, my_parallel_data,
          my_task_data, return_address);
    }
    if (ompt_enabled.ompt_callback_sync_region) {
      ompt_callbacks.ompt_callback(ompt_callback_sync_region)(
          ompt_sync_region_taskwait, ompt_scope_end, my_parallel_data,
          my_task_data, return_address);
    }
    taskdata->ompt_task_info.frame.reenter_runtime_frame = NULL;
#if KMP_USE_TASK_AFFINITY && KMP_TASK_AFFINITY_MEASURE_TIME
    start_task_execution_measurement(taskdata);
#endif

    ANNOTATE_HAPPENS_AFTER(taskdata);
  }

  KA_TRACE(10, ("__kmpc_omp_taskwait(exit): T#%d task %p finished waiting, "
                "returning TASK_CURRENT_NOT_QUEUED\n",
                gtid, taskdata));

  return TASK_CURRENT_NOT_QUEUED;
}
#endif // OMPT_SUPPORT && OMPT_OPTIONAL

// __kmpc_omp_taskwait: Wait until all tasks generated by the current task are
// complete
kmp_int32 __kmpc_omp_taskwait(ident_t *loc_ref, kmp_int32 gtid) {
#if OMPT_SUPPORT && OMPT_OPTIONAL
  if (UNLIKELY(ompt_enabled.enabled)) {
    return __ompt_enabled_taskwait(loc_ref, gtid,
        OMPT_GET_FRAME_ADDRESS(1), OMPT_GET_RETURN_ADDRESS(0));
  }
#endif
  kmp_taskdata_t *taskdata;
  kmp_info_t *thread;
  int thread_finished = FALSE;
  KMP_SET_THREAD_STATE_BLOCK(TASKWAIT);

  KA_TRACE(10, ("__kmpc_omp_taskwait(enter): T#%d loc=%p\n", gtid, loc_ref));

  if (__kmp_tasking_mode != tskm_immediate_exec) {
    thread = __kmp_threads[gtid];
    taskdata = thread->th.th_current_task;
#if KMP_USE_TASK_AFFINITY && KMP_TASK_AFFINITY_MEASURE_TIME
  stop_task_execution_measurement(taskdata);
#endif

// Debugger: The taskwait is active. Store location and thread encountered the
// taskwait.
#if USE_ITT_BUILD
// Note: These values are used by ITT events as well.
#endif /* USE_ITT_BUILD */
    taskdata->td_taskwait_counter += 1;
    taskdata->td_taskwait_ident = loc_ref;
    taskdata->td_taskwait_thread = gtid + 1;

#if USE_ITT_BUILD
    void *itt_sync_obj = __kmp_itt_taskwait_object(gtid);
    if (itt_sync_obj != NULL)
      __kmp_itt_taskwait_starting(gtid, itt_sync_obj);
#endif /* USE_ITT_BUILD */

    bool must_wait =
        !taskdata->td_flags.team_serial && !taskdata->td_flags.final;

#if OMP_45_ENABLED
    must_wait = must_wait || (thread->th.th_task_team != NULL &&
                              thread->th.th_task_team->tt.tt_found_proxy_tasks);
#endif
    if (must_wait) {
      kmp_flag_32 flag(
          RCAST(volatile kmp_uint32 *, &taskdata->td_incomplete_child_tasks),
          0U);
      while (TCR_4(taskdata->td_incomplete_child_tasks) != 0) {
        flag.execute_tasks(thread, gtid, FALSE,
                           &thread_finished USE_ITT_BUILD_ARG(itt_sync_obj),
                           __kmp_task_stealing_constraint);
      }
    }
#if USE_ITT_BUILD
    if (itt_sync_obj != NULL)
      __kmp_itt_taskwait_finished(gtid, itt_sync_obj);
#endif /* USE_ITT_BUILD */

    // Debugger:  The taskwait is completed. Location remains, but thread is
    // negated.
    taskdata->td_taskwait_thread = -taskdata->td_taskwait_thread;
#if KMP_USE_TASK_AFFINITY && KMP_TASK_AFFINITY_MEASURE_TIME
  start_task_execution_measurement(taskdata);
#endif
    ANNOTATE_HAPPENS_AFTER(taskdata);
  }

  KA_TRACE(10, ("__kmpc_omp_taskwait(exit): T#%d task %p finished waiting, "
                "returning TASK_CURRENT_NOT_QUEUED\n",
                gtid, taskdata));

  return TASK_CURRENT_NOT_QUEUED;
}

// __kmpc_omp_taskyield: switch to a different task
kmp_int32 __kmpc_omp_taskyield(ident_t *loc_ref, kmp_int32 gtid, int end_part) {
  kmp_taskdata_t *taskdata;
  kmp_info_t *thread;
  int thread_finished = FALSE;

  KMP_COUNT_BLOCK(OMP_TASKYIELD);
  KMP_SET_THREAD_STATE_BLOCK(TASKYIELD);

  KA_TRACE(10, ("__kmpc_omp_taskyield(enter): T#%d loc=%p end_part = %d\n",
                gtid, loc_ref, end_part));

  if (__kmp_tasking_mode != tskm_immediate_exec && __kmp_init_parallel) {
    thread = __kmp_threads[gtid];
    taskdata = thread->th.th_current_task;
// Should we model this as a task wait or not?
// Debugger: The taskwait is active. Store location and thread encountered the
// taskwait.
#if USE_ITT_BUILD
// Note: These values are used by ITT events as well.
#endif /* USE_ITT_BUILD */
    taskdata->td_taskwait_counter += 1;
    taskdata->td_taskwait_ident = loc_ref;
    taskdata->td_taskwait_thread = gtid + 1;

#if USE_ITT_BUILD
    void *itt_sync_obj = __kmp_itt_taskwait_object(gtid);
    if (itt_sync_obj != NULL)
      __kmp_itt_taskwait_starting(gtid, itt_sync_obj);
#endif /* USE_ITT_BUILD */
    if (!taskdata->td_flags.team_serial) {
      kmp_task_team_t *task_team = thread->th.th_task_team;
      if (task_team != NULL) {
        if (KMP_TASKING_ENABLED(task_team)) {
#if OMPT_SUPPORT
          if (__builtin_expect(ompt_enabled.enabled,0))
            thread->th.ompt_thread_info.ompt_task_yielded = 1;
#endif
          __kmp_execute_tasks_32(
              thread, gtid, NULL, FALSE,
              &thread_finished USE_ITT_BUILD_ARG(itt_sync_obj),
              __kmp_task_stealing_constraint);
#if OMPT_SUPPORT
          if (__builtin_expect(ompt_enabled.enabled,0))
            thread->th.ompt_thread_info.ompt_task_yielded = 0;
#endif
        }
      }
    }
#if USE_ITT_BUILD
    if (itt_sync_obj != NULL)
      __kmp_itt_taskwait_finished(gtid, itt_sync_obj);
#endif /* USE_ITT_BUILD */

    // Debugger:  The taskwait is completed. Location remains, but thread is
    // negated.
    taskdata->td_taskwait_thread = -taskdata->td_taskwait_thread;
  }

  KA_TRACE(10, ("__kmpc_omp_taskyield(exit): T#%d task %p resuming, "
                "returning TASK_CURRENT_NOT_QUEUED\n",
                gtid, taskdata));

  return TASK_CURRENT_NOT_QUEUED;
}

// TODO: change to OMP_50_ENABLED, need to change build tools for this to work
#if OMP_45_ENABLED
// Task Reduction implementation

typedef struct kmp_task_red_flags {
  unsigned lazy_priv : 1; // hint: (1) use lazy allocation (big objects)
  unsigned reserved31 : 31;
} kmp_task_red_flags_t;

// internal structure for reduction data item related info
typedef struct kmp_task_red_data {
  void *reduce_shar; // shared reduction item
  size_t reduce_size; // size of data item
  void *reduce_priv; // thread specific data
  void *reduce_pend; // end of private data for comparison op
  void *reduce_init; // data initialization routine
  void *reduce_fini; // data finalization routine
  void *reduce_comb; // data combiner routine
  kmp_task_red_flags_t flags; // flags for additional info from compiler
} kmp_task_red_data_t;

// structure sent us by compiler - one per reduction item
typedef struct kmp_task_red_input {
  void *reduce_shar; // shared reduction item
  size_t reduce_size; // size of data item
  void *reduce_init; // data initialization routine
  void *reduce_fini; // data finalization routine
  void *reduce_comb; // data combiner routine
  kmp_task_red_flags_t flags; // flags for additional info from compiler
} kmp_task_red_input_t;

/*!
@ingroup TASKING
@param gtid      Global thread ID
@param num       Number of data items to reduce
@param data      Array of data for reduction
@return The taskgroup identifier

Initialize task reduction for the taskgroup.
*/
void *__kmpc_task_reduction_init(int gtid, int num, void *data) {
  kmp_info_t *thread = __kmp_threads[gtid];
  kmp_taskgroup_t *tg = thread->th.th_current_task->td_taskgroup;
  kmp_int32 nth = thread->th.th_team_nproc;
  kmp_task_red_input_t *input = (kmp_task_red_input_t *)data;
  kmp_task_red_data_t *arr;

  // check input data just in case
  KMP_ASSERT(tg != NULL);
  KMP_ASSERT(data != NULL);
  KMP_ASSERT(num > 0);
  if (nth == 1) {
    KA_TRACE(10, ("__kmpc_task_reduction_init: T#%d, tg %p, exiting nth=1\n",
                  gtid, tg));
    return (void *)tg;
  }
  KA_TRACE(10, ("__kmpc_task_reduction_init: T#%d, taskgroup %p, #items %d\n",
                gtid, tg, num));
  arr = (kmp_task_red_data_t *)__kmp_thread_malloc(
      thread, num * sizeof(kmp_task_red_data_t));
  for (int i = 0; i < num; ++i) {
    void (*f_init)(void *) = (void (*)(void *))(input[i].reduce_init);
    size_t size = input[i].reduce_size - 1;
    // round the size up to cache line per thread-specific item
    size += CACHE_LINE - size % CACHE_LINE;
    KMP_ASSERT(input[i].reduce_comb != NULL); // combiner is mandatory
    arr[i].reduce_shar = input[i].reduce_shar;
    arr[i].reduce_size = size;
    arr[i].reduce_init = input[i].reduce_init;
    arr[i].reduce_fini = input[i].reduce_fini;
    arr[i].reduce_comb = input[i].reduce_comb;
    arr[i].flags = input[i].flags;
    if (!input[i].flags.lazy_priv) {
      // allocate cache-line aligned block and fill it with zeros
      arr[i].reduce_priv = __kmp_allocate(nth * size);
      arr[i].reduce_pend = (char *)(arr[i].reduce_priv) + nth * size;
      if (f_init != NULL) {
        // initialize thread-specific items
        for (int j = 0; j < nth; ++j) {
          f_init((char *)(arr[i].reduce_priv) + j * size);
        }
      }
    } else {
      // only allocate space for pointers now,
      // objects will be lazily allocated/initialized once requested
      arr[i].reduce_priv = __kmp_allocate(nth * sizeof(void *));
    }
  }
  tg->reduce_data = (void *)arr;
  tg->reduce_num_data = num;
  return (void *)tg;
}

/*!
@ingroup TASKING
@param gtid    Global thread ID
@param tskgrp  The taskgroup ID (optional)
@param data    Shared location of the item
@return The pointer to per-thread data

Get thread-specific location of data item
*/
void *__kmpc_task_reduction_get_th_data(int gtid, void *tskgrp, void *data) {
  kmp_info_t *thread = __kmp_threads[gtid];
  kmp_int32 nth = thread->th.th_team_nproc;
  if (nth == 1)
    return data; // nothing to do

  kmp_taskgroup_t *tg = (kmp_taskgroup_t *)tskgrp;
  if (tg == NULL)
    tg = thread->th.th_current_task->td_taskgroup;
  KMP_ASSERT(tg != NULL);
  kmp_task_red_data_t *arr = (kmp_task_red_data_t *)(tg->reduce_data);
  kmp_int32 num = tg->reduce_num_data;
  kmp_int32 tid = thread->th.th_info.ds.ds_tid;

  KMP_ASSERT(data != NULL);
  while (tg != NULL) {
    for (int i = 0; i < num; ++i) {
      if (!arr[i].flags.lazy_priv) {
        if (data == arr[i].reduce_shar ||
            (data >= arr[i].reduce_priv && data < arr[i].reduce_pend))
          return (char *)(arr[i].reduce_priv) + tid * arr[i].reduce_size;
      } else {
        // check shared location first
        void **p_priv = (void **)(arr[i].reduce_priv);
        if (data == arr[i].reduce_shar)
          goto found;
        // check if we get some thread specific location as parameter
        for (int j = 0; j < nth; ++j)
          if (data == p_priv[j])
            goto found;
        continue; // not found, continue search
      found:
        if (p_priv[tid] == NULL) {
          // allocate thread specific object lazily
          void (*f_init)(void *) = (void (*)(void *))(arr[i].reduce_init);
          p_priv[tid] = __kmp_allocate(arr[i].reduce_size);
          if (f_init != NULL) {
            f_init(p_priv[tid]);
          }
        }
        return p_priv[tid];
      }
    }
    tg = tg->parent;
    arr = (kmp_task_red_data_t *)(tg->reduce_data);
    num = tg->reduce_num_data;
  }
  KMP_ASSERT2(0, "Unknown task reduction item");
  return NULL; // ERROR, this line never executed
}

// Finalize task reduction.
// Called from __kmpc_end_taskgroup()
static void __kmp_task_reduction_fini(kmp_info_t *th, kmp_taskgroup_t *tg) {
  kmp_int32 nth = th->th.th_team_nproc;
  KMP_DEBUG_ASSERT(nth > 1); // should not be called if nth == 1
  kmp_task_red_data_t *arr = (kmp_task_red_data_t *)tg->reduce_data;
  kmp_int32 num = tg->reduce_num_data;
  for (int i = 0; i < num; ++i) {
    void *sh_data = arr[i].reduce_shar;
    void (*f_fini)(void *) = (void (*)(void *))(arr[i].reduce_fini);
    void (*f_comb)(void *, void *) =
        (void (*)(void *, void *))(arr[i].reduce_comb);
    if (!arr[i].flags.lazy_priv) {
      void *pr_data = arr[i].reduce_priv;
      size_t size = arr[i].reduce_size;
      for (int j = 0; j < nth; ++j) {
        void *priv_data = (char *)pr_data + j * size;
        f_comb(sh_data, priv_data); // combine results
        if (f_fini)
          f_fini(priv_data); // finalize if needed
      }
    } else {
      void **pr_data = (void **)(arr[i].reduce_priv);
      for (int j = 0; j < nth; ++j) {
        if (pr_data[j] != NULL) {
          f_comb(sh_data, pr_data[j]); // combine results
          if (f_fini)
            f_fini(pr_data[j]); // finalize if needed
          __kmp_free(pr_data[j]);
        }
      }
    }
    __kmp_free(arr[i].reduce_priv);
  }
  __kmp_thread_free(th, arr);
  tg->reduce_data = NULL;
  tg->reduce_num_data = 0;
}
#endif

#if OMP_40_ENABLED
// __kmpc_taskgroup: Start a new taskgroup
void __kmpc_taskgroup(ident_t *loc, int gtid) {
  kmp_info_t *thread = __kmp_threads[gtid];
  kmp_taskdata_t *taskdata = thread->th.th_current_task;
  kmp_taskgroup_t *tg_new =
      (kmp_taskgroup_t *)__kmp_thread_malloc(thread, sizeof(kmp_taskgroup_t));
  KA_TRACE(10, ("__kmpc_taskgroup: T#%d loc=%p group=%p\n", gtid, loc, tg_new));
  tg_new->count = 0;
  tg_new->cancel_request = cancel_noreq;
  tg_new->parent = taskdata->td_taskgroup;
// TODO: change to OMP_50_ENABLED, need to change build tools for this to work
#if OMP_45_ENABLED
  tg_new->reduce_data = NULL;
  tg_new->reduce_num_data = 0;
#endif
  taskdata->td_taskgroup = tg_new;

#if OMPT_SUPPORT && OMPT_OPTIONAL
  if (__builtin_expect(ompt_enabled.ompt_callback_sync_region,0)) {
    void * codeptr = OMPT_LOAD_RETURN_ADDRESS(gtid);
    if (!codeptr) codeptr = OMPT_GET_RETURN_ADDRESS(0);
    kmp_team_t *team = thread->th.th_team;
    ompt_data_t my_task_data = taskdata->ompt_task_info.task_data;
    // FIXME: I think this is wrong for lwt!
    ompt_data_t my_parallel_data = team->t.ompt_team_info.parallel_data;

    ompt_callbacks.ompt_callback(ompt_callback_sync_region)(
        ompt_sync_region_taskgroup, ompt_scope_begin, &(my_parallel_data),
        &(my_task_data), codeptr);
  }
#endif
}

// __kmpc_end_taskgroup: Wait until all tasks generated by the current task
//                       and its descendants are complete
void __kmpc_end_taskgroup(ident_t *loc, int gtid) {
  kmp_info_t *thread = __kmp_threads[gtid];
  kmp_taskdata_t *taskdata = thread->th.th_current_task;
  kmp_taskgroup_t *taskgroup = taskdata->td_taskgroup;
  int thread_finished = FALSE;

#if OMPT_SUPPORT && OMPT_OPTIONAL
  kmp_team_t *team;
  ompt_data_t my_task_data;
  ompt_data_t my_parallel_data;
  void * codeptr;
  if (__builtin_expect(ompt_enabled.enabled,0))
  {
    team = thread->th.th_team;
    my_task_data = taskdata->ompt_task_info.task_data;
    // FIXME: I think this is wrong for lwt!
    my_parallel_data = team->t.ompt_team_info.parallel_data;
    codeptr = OMPT_LOAD_RETURN_ADDRESS(gtid);
    if (!codeptr) codeptr = OMPT_GET_RETURN_ADDRESS(0);
  }
#endif

  KA_TRACE(10, ("__kmpc_end_taskgroup(enter): T#%d loc=%p\n", gtid, loc));
  KMP_DEBUG_ASSERT(taskgroup != NULL);
  KMP_SET_THREAD_STATE_BLOCK(TASKGROUP);

  if (__kmp_tasking_mode != tskm_immediate_exec) {
#if USE_ITT_BUILD
    // For ITT the taskgroup wait is similar to taskwait until we need to
    // distinguish them
    void *itt_sync_obj = __kmp_itt_taskwait_object(gtid);
    if (itt_sync_obj != NULL)
      __kmp_itt_taskwait_starting(gtid, itt_sync_obj);
#endif /* USE_ITT_BUILD */

#if OMPT_SUPPORT && OMPT_OPTIONAL
    if (__builtin_expect(ompt_enabled.ompt_callback_sync_region_wait,0)) {
      ompt_callbacks.ompt_callback(ompt_callback_sync_region_wait)(
          ompt_sync_region_taskgroup,
          ompt_scope_begin,
          &(my_parallel_data),
          &(my_task_data),
          codeptr);
    }
#endif

#if OMP_45_ENABLED
    if (!taskdata->td_flags.team_serial ||
        (thread->th.th_task_team != NULL &&
         thread->th.th_task_team->tt.tt_found_proxy_tasks))
#else
    if (!taskdata->td_flags.team_serial)
#endif
    {
      kmp_flag_32 flag(RCAST(kmp_uint32 *, &taskgroup->count), 0U);
      while (TCR_4(taskgroup->count) != 0) {
        flag.execute_tasks(thread, gtid, FALSE,
                           &thread_finished USE_ITT_BUILD_ARG(itt_sync_obj),
                           __kmp_task_stealing_constraint);
      }
    }

#if OMPT_SUPPORT && OMPT_OPTIONAL
    if (__builtin_expect(ompt_enabled.ompt_callback_sync_region_wait,0)) {
        ompt_callbacks.ompt_callback(ompt_callback_sync_region_wait)(
            ompt_sync_region_taskgroup,
            ompt_scope_end,
            &(my_parallel_data),
            &(my_task_data),
            codeptr);
    }
#endif

#if USE_ITT_BUILD
    if (itt_sync_obj != NULL)
      __kmp_itt_taskwait_finished(gtid, itt_sync_obj);
#endif /* USE_ITT_BUILD */
  }
  KMP_DEBUG_ASSERT(taskgroup->count == 0);

// TODO: change to OMP_50_ENABLED, need to change build tools for this to work
#if OMP_45_ENABLED
  if (taskgroup->reduce_data != NULL) // need to reduce?
    __kmp_task_reduction_fini(thread, taskgroup);
#endif
  // Restore parent taskgroup for the current task
  taskdata->td_taskgroup = taskgroup->parent;
  __kmp_thread_free(thread, taskgroup);

  KA_TRACE(10, ("__kmpc_end_taskgroup(exit): T#%d task %p finished waiting\n",
                gtid, taskdata));
  ANNOTATE_HAPPENS_AFTER(taskdata);

#if OMPT_SUPPORT && OMPT_OPTIONAL
  if (__builtin_expect(ompt_enabled.ompt_callback_sync_region,0)) {
    ompt_callbacks.ompt_callback(ompt_callback_sync_region)(
        ompt_sync_region_taskgroup, ompt_scope_end, &(my_parallel_data),
        &(my_task_data), codeptr);
  }
#endif
}
#endif

// __kmp_remove_my_task: remove a task from my own deque
static kmp_task_t *__kmp_remove_my_task(kmp_info_t *thread, kmp_int32 gtid,
                                        kmp_task_team_t *task_team,
                                        kmp_int32 is_constrained) {
  kmp_task_t *task;
  kmp_taskdata_t *taskdata;
  kmp_thread_data_t *thread_data;
  kmp_uint32 tail;

  KMP_DEBUG_ASSERT(__kmp_tasking_mode != tskm_immediate_exec);
  KMP_DEBUG_ASSERT(task_team->tt.tt_threads_data !=
                   NULL); // Caller should check this condition

  thread_data = &task_team->tt.tt_threads_data[__kmp_tid_from_gtid(gtid)];

#if KMP_USE_TASK_AFFINITY
  // Allocate deque if necessary
  //kmp_info_t *thread = __kmp_threads[gtid];
  // No lock needed since only owner can allocate
  if (thread_data->td.td_deque == NULL) {
    __kmp_alloc_task_deque(thread, thread_data);
  }
#endif

  KA_TRACE(10, ("__kmp_remove_my_task(enter): T#%d ntasks=%d head=%u tail=%u\n",
                gtid, thread_data->td.td_deque_ntasks,
                thread_data->td.td_deque_head, thread_data->td.td_deque_tail));

  if (TCR_4(thread_data->td.td_deque_ntasks) == 0) {
    KA_TRACE(10,
             ("__kmp_remove_my_task(exit #1): T#%d No tasks to remove: "
              "ntasks=%d head=%u tail=%u\n",
              gtid, thread_data->td.td_deque_ntasks,
              thread_data->td.td_deque_head, thread_data->td.td_deque_tail));
    return NULL;
  }

  __kmp_acquire_bootstrap_lock(&thread_data->td.td_deque_lock);

  if (TCR_4(thread_data->td.td_deque_ntasks) == 0) {
    __kmp_release_bootstrap_lock(&thread_data->td.td_deque_lock);
    KA_TRACE(10,
             ("__kmp_remove_my_task(exit #2): T#%d No tasks to remove: "
              "ntasks=%d head=%u tail=%u\n",
              gtid, thread_data->td.td_deque_ntasks,
              thread_data->td.td_deque_head, thread_data->td.td_deque_tail));
    return NULL;
  }

  tail = (thread_data->td.td_deque_tail - 1) &
         TASK_DEQUE_MASK(thread_data->td); // Wrap index.
  taskdata = thread_data->td.td_deque[tail];

  if (is_constrained && (taskdata->td_flags.tiedness == TASK_TIED)) {
    // we need to check if the candidate obeys task scheduling constraint:
    // only child of current task can be scheduled
    kmp_taskdata_t *current = thread->th.th_current_task;
    kmp_int32 level = current->td_level;
    kmp_taskdata_t *parent = taskdata->td_parent;
    // KA_TRACE(10, ("TASK AFFINITY: T#%d Check parent level ntasks=%d head=%u tail=%u -- curtask:%p curlvl:%d ptask:%p plvl:%d\n",
    //             gtid, thread_data->td.td_deque_ntasks,
    //             thread_data->td.td_deque_head, thread_data->td.td_deque_tail, current, level, parent, parent->td_level));

    while (parent != current && parent->td_level > level) {
      parent = parent->td_parent; // check generation up to the level of the
      KA_TRACE(10, ("TASK AFFINITY: T#%d Check parent level ntasks=%d head=%u tail=%u -- curtask:%p curlvl:%d ptask:%p plvl:%d\n",
                gtid, thread_data->td.td_deque_ntasks,
                thread_data->td.td_deque_head, thread_data->td.td_deque_tail, current, level, parent, parent->td_level));
      // current task
      KMP_DEBUG_ASSERT(parent != NULL);
    }
    if (parent != current) {
      // If the tail task is not a child, then no other child can appear in the
      // deque.
      __kmp_release_bootstrap_lock(&thread_data->td.td_deque_lock);
      KA_TRACE(10,
               ("__kmp_remove_my_task(exit #2): T#%d No tasks to remove: "
                "ntasks=%d head=%u tail=%u\n",
                gtid, thread_data->td.td_deque_ntasks,
                thread_data->td.td_deque_head, thread_data->td.td_deque_tail));
      return NULL;
    }
  }

  thread_data->td.td_deque_tail = tail;
  TCW_4(thread_data->td.td_deque_ntasks, thread_data->td.td_deque_ntasks - 1);

  __kmp_release_bootstrap_lock(&thread_data->td.td_deque_lock);

  KA_TRACE(10, ("__kmp_remove_my_task(exit #2): T#%d task %p removed: "
                "ntasks=%d head=%u tail=%u\n",
                gtid, taskdata, thread_data->td.td_deque_ntasks,
                thread_data->td.td_deque_head, thread_data->td.td_deque_tail));

  task = KMP_TASKDATA_TO_TASK(taskdata);
  return task;
}

// __kmp_steal_task: remove a task from another thread's deque
// Assume that calling thread has already checked existence of
// task_team thread_data before calling this routine.
static kmp_task_t *__kmp_steal_task(kmp_info_t *victim, kmp_int32 gtid,
                                    kmp_task_team_t *task_team,
                                    volatile kmp_int32 *unfinished_threads,
                                    int *thread_finished,
                                    kmp_int32 is_constrained) {
  kmp_task_t *task;
  kmp_taskdata_t *taskdata;
  kmp_thread_data_t *victim_td, *threads_data;
  kmp_int32 victim_tid;

  KMP_DEBUG_ASSERT(__kmp_tasking_mode != tskm_immediate_exec);

  threads_data = task_team->tt.tt_threads_data;
  KMP_DEBUG_ASSERT(threads_data != NULL); // Caller should check this condition

  victim_tid = victim->th.th_info.ds.ds_tid;
  victim_td = &threads_data[victim_tid];

  KA_TRACE(10, ("__kmp_steal_task(enter): T#%d try to steal from T#%d: "
                "task_team=%p ntasks=%d "
                "head=%u tail=%u\n",
                gtid, __kmp_gtid_from_thread(victim), task_team,
                victim_td->td.td_deque_ntasks, victim_td->td.td_deque_head,
                victim_td->td.td_deque_tail));

  if ((TCR_4(victim_td->td.td_deque_ntasks) ==
       0) || // Caller should not check this condition
      (TCR_PTR(victim->th.th_task_team) !=
       task_team)) // GEH: why would this happen?
  {
    KA_TRACE(10, ("__kmp_steal_task(exit #1): T#%d could not steal from T#%d: "
                  "task_team=%p "
                  "ntasks=%d head=%u tail=%u\n",
                  gtid, __kmp_gtid_from_thread(victim), task_team,
                  victim_td->td.td_deque_ntasks, victim_td->td.td_deque_head,
                  victim_td->td.td_deque_tail));
    return NULL;
  }

  __kmp_acquire_bootstrap_lock(&victim_td->td.td_deque_lock);

  // Check again after we acquire the lock
  if ((TCR_4(victim_td->td.td_deque_ntasks) == 0) ||
      (TCR_PTR(victim->th.th_task_team) !=
       task_team)) // GEH: why would this happen?
  {
    __kmp_release_bootstrap_lock(&victim_td->td.td_deque_lock);
    KA_TRACE(10, ("__kmp_steal_task(exit #2): T#%d could not steal from T#%d: "
                  "task_team=%p "
                  "ntasks=%d head=%u tail=%u\n",
                  gtid, __kmp_gtid_from_thread(victim), task_team,
                  victim_td->td.td_deque_ntasks, victim_td->td.td_deque_head,
                  victim_td->td.td_deque_tail));
    return NULL;
  }

  KMP_DEBUG_ASSERT(victim_td->td.td_deque != NULL);

  taskdata = victim_td->td.td_deque[victim_td->td.td_deque_head];

  if (is_constrained) {
    // we need to check if the candidate obeys task scheduling constraint:
    // only descendant of current task can be scheduled
    kmp_taskdata_t *current = __kmp_threads[gtid]->th.th_current_task;
    kmp_int32 level = current->td_level;
    kmp_taskdata_t *parent = taskdata->td_parent;

    while (parent != current && parent->td_level > level) {
      parent = parent->td_parent; // check generation up to the level of the
      // current task
      KMP_DEBUG_ASSERT(parent != NULL);
    }
    if (parent != current) {
      // If the head task is not a descendant of the current task then do not
      // steal it. No other task in victim's deque can be a descendant of the
      // current task.
      __kmp_release_bootstrap_lock(&victim_td->td.td_deque_lock);
      KA_TRACE(10, ("__kmp_steal_task(exit #2): T#%d could not steal from "
                    "T#%d: task_team=%p "
                    "ntasks=%d head=%u tail=%u\n",
                    gtid,
                    __kmp_gtid_from_thread(threads_data[victim_tid].td.td_thr),
                    task_team, victim_td->td.td_deque_ntasks,
                    victim_td->td.td_deque_head, victim_td->td.td_deque_tail));
      return NULL;
    }
  }
  // Bump head pointer and Wrap.
  victim_td->td.td_deque_head =
      (victim_td->td.td_deque_head + 1) & TASK_DEQUE_MASK(victim_td->td);
  if (*thread_finished) {
    // We need to un-mark this victim as a finished victim.  This must be done
    // before releasing the lock, or else other threads (starting with the
    // master victim) might be prematurely released from the barrier!!!
    kmp_int32 count;

    count = KMP_TEST_THEN_INC32(unfinished_threads);

    KA_TRACE(
        20,
        ("__kmp_steal_task: T#%d inc unfinished_threads to %d: task_team=%p\n",
         gtid, count + 1, task_team));

    *thread_finished = FALSE;
  }
  TCW_4(victim_td->td.td_deque_ntasks,
        TCR_4(victim_td->td.td_deque_ntasks) - 1);

  __kmp_release_bootstrap_lock(&victim_td->td.td_deque_lock);

  KMP_COUNT_BLOCK(TASK_stolen);
  KA_TRACE(
      10,
      ("__kmp_steal_task(exit #3): T#%d stole task %p from T#%d: task_team=%p "
       "ntasks=%d head=%u tail=%u\n",
       gtid, taskdata, __kmp_gtid_from_thread(victim), task_team,
       victim_td->td.td_deque_ntasks, victim_td->td.td_deque_head,
       victim_td->td.td_deque_tail));

  if(taskdata == NULL)
    return NULL;
  task = KMP_TASKDATA_TO_TASK(taskdata);
  return task;
}

// __kmp_execute_tasks_template: Choose and execute tasks until either the
// condition is statisfied (return true) or there are none left (return false).
//
// final_spin is TRUE if this is the spin at the release barrier.
// thread_finished indicates whether the thread is finished executing all
// the tasks it has on its deque, and is at the release barrier.
// spinner is the location on which to spin.
// spinner == NULL means only execute a single task and return.
// checker is the value to check to terminate the spin.
template <class C>
static inline int __kmp_execute_tasks_template(
    kmp_info_t *thread, kmp_int32 gtid, C *flag, int final_spin,
    int *thread_finished USE_ITT_BUILD_ARG(void *itt_sync_obj),
    kmp_int32 is_constrained) {
  kmp_task_team_t *task_team = thread->th.th_task_team;
  kmp_thread_data_t *threads_data;
  kmp_task_t *task;
  kmp_info_t *other_thread;
  kmp_info_t *numa_thread;
  kmp_taskdata_t *current_task = thread->th.th_current_task;
  volatile kmp_int32 *unfinished_threads;
  kmp_int32 nthreads, victim = -2, use_own_tasks = 1, new_victim = 0,
                      tid = thread->th.th_info.ds.ds_tid;

  KMP_DEBUG_ASSERT(__kmp_tasking_mode != tskm_immediate_exec);
  KMP_DEBUG_ASSERT(thread == __kmp_threads[gtid]);

  if (task_team == NULL)
    return FALSE;

  KA_TRACE(15, ("__kmp_execute_tasks_template(enter): T#%d final_spin=%d "
                "*thread_finished=%d current_task=%p\n",
                gtid, final_spin, *thread_finished, thread->th.th_current_task));

  thread->th.th_reap_state = KMP_NOT_SAFE_TO_REAP;
  threads_data = (kmp_thread_data_t *)TCR_PTR(task_team->tt.tt_threads_data);
  KMP_DEBUG_ASSERT(threads_data != NULL);

  nthreads = task_team->tt.tt_nproc;
  unfinished_threads = &(task_team->tt.tt_unfinished_threads);
#if OMP_45_ENABLED
  KMP_DEBUG_ASSERT(nthreads > 1 || task_team->tt.tt_found_proxy_tasks);
#else
  KMP_DEBUG_ASSERT(nthreads > 1);
#endif
  KMP_DEBUG_ASSERT(TCR_4(*unfinished_threads) >= 0); // TODO: Data Race????

#if KMP_USE_TASK_AFFINITY
  bool last_stolen_from_was_numa_thread = false;
  int last_stolen_numa_thread = -1;
  int max_numa_tries_before_normal = KMP_TASK_AFFINITY_MAX_NUM_STEAL_TRIES;
  int count_numa_tries = 0;
#endif

  while (1) { // Outer loop keeps trying to find tasks in case of single thread
    // getting tasks from target constructs
    while (1) { // Inner loop to find a task and execute it
      task = NULL;
      if (use_own_tasks) { // check on own queue first
        task = __kmp_remove_my_task(thread, gtid, task_team, is_constrained);
      }
      if ((task == NULL) && (nthreads > 1)) { // Steal a task
        int asleep = 1;
        use_own_tasks = 0;
        // Try to steal from the last place I stole from successfully.
        if (victim == -2) { // haven't stolen anything yet
          victim = threads_data[tid].td.td_deque_last_stolen;
          if (victim !=
              -1) // if we have a last stolen from victim, get the thread
            other_thread = threads_data[victim].td.td_thr;
        }
        if (victim != -1) { // found last victim
          asleep = 0;
        } else if (!new_victim) { // no recent steals and we haven't already
          // used a new victim; select a random thread
          do { // Find a different thread to steal work from.
            // Pick a random thread. Initial plan was to cycle through all the
            // threads, and only return if we tried to steal from every thread,
            // and failed.  Arch says that's not such a great idea.
            victim = __kmp_get_random(thread) % (nthreads - 1);
            if (victim >= tid) {
              ++victim; // Adjusts random distribution to exclude self
            }
            // Found a potential victim
            other_thread = threads_data[victim].td.td_thr;
            // There is a slight chance that __kmp_enable_tasking() did not wake
            // up all threads waiting at the barrier.  If victim is sleeping,
            // then wake it up. Since we were going to pay the cache miss
            // penalty for referencing another thread's kmp_info_t struct
            // anyway,
            // the check shouldn't cost too much performance at this point. In
            // extra barrier mode, tasks do not sleep at the separate tasking
            // barrier, so this isn't a problem.
            asleep = 0;
            if ((__kmp_tasking_mode == tskm_task_teams) &&
                (__kmp_dflt_blocktime != KMP_MAX_BLOCKTIME) &&
                (TCR_PTR(CCAST(void *, other_thread->th.th_sleep_loc)) !=
                 NULL)) {
              asleep = 1;
              __kmp_null_resume_wrapper(__kmp_gtid_from_thread(other_thread),
                                        other_thread->th.th_sleep_loc);
              // A sleeping thread should not have any tasks on it's queue.
              // There is a slight possibility that it resumes, steals a task
              // from another thread, which spawns more tasks, all in the time
              // that it takes this thread to check => don't write an assertion
              // that the victim's queue is empty.  Try stealing from a
              // different thread.
            }
          } while (asleep);
        }

        if (!asleep) {
#if KMP_USE_TASK_AFFINITY && KMP_TASK_AFFINITY_NUMA_STEALING_ENABLED
          int tmp_victim_tid = -1;
          int tmp_victim_gtid = -1;
          double time1=0, time2=0;
          //time1 = 0;
          //time2 = 0;

          task = NULL;
          // also check if task team holds tasks with affinity.. otherwise do not use numa aware stealing
          //volatile int * num_aff_tasks = &(task_team->tt.tt_num_tasks_with_aff);
          //if(enable_numa_aware_stealing && task_team->tt.tt_numa_domains_set && *num_aff_tasks > 0) {
          if(enable_numa_aware_stealing && task_team->tt.tt_numa_domains_set) {
#if KMP_TASK_AFFINITY_MEASURE_TIME
            time1 = get_wall_time2();
#endif
            if(last_stolen_from_was_numa_thread) {
              // remember last choice of successful steal in numa domain
              // TODO: change that and save it in threads struct
              tmp_victim_gtid = last_stolen_numa_thread;
            } else {
              int my_domain = thread->th.th_task_aff_my_domain_nr;
              // fprintf(stderr, "NUMA steal: T#%d my domain = %d\n", gtid, my_domain);
              //srand(time(NULL));
              // int tmp_rand_idx = rand() % task_team->tt.tt_numa_domain_size[my_domain]; // numa_domain_size[my_domain];
              int tmp_rand_idx = __kmp_get_random(thread) % task_team->tt.tt_numa_domain_size[my_domain];
              tmp_victim_gtid = task_team->tt.tt_map_threads_in_domain[my_domain][tmp_rand_idx]; //map_threads_in_numa_domain[my_domain][tmp_rand_idx];
              // int tmp_rand_idx = rand() % numa_domain_size[my_domain];
              // fprintf(stderr, "NUMA steal: T#%d rand idx = %d\n", gtid, tmp_rand_idx);
              // tmp_victim_gtid = map_threads_in_numa_domain[my_domain][tmp_rand_idx]; //;
              // fprintf(stderr, "NUMA steal: T#%d victim gtid = %d\n", gtid, tmp_victim_gtid);
              if(tmp_victim_gtid == gtid) {
                // exclude self
                tmp_rand_idx = ++tmp_rand_idx % task_team->tt.tt_numa_domain_size[my_domain]; //numa_domain_size[my_domain];
                tmp_victim_gtid = task_team->tt.tt_map_threads_in_domain[my_domain][tmp_rand_idx]; //map_threads_in_numa_domain[my_domain][tmp_rand_idx];

                // tmp_rand_idx = ++tmp_rand_idx % numa_domain_size[my_domain];
                // tmp_victim_gtid = map_threads_in_numa_domain[my_domain][tmp_rand_idx];
                // fprintf(stderr, "NUMA steal: T#%d new victim gtid = %d\n", gtid, tmp_victim_gtid);
              }
              // tmp_victim_tid = __kmp_tid_from_gtid(tmp_victim_gtid);
              // fprintf(stderr, "NUMA steal: T#%d new victim tid = %d\n", gtid, tmp_victim_tid);
              // get thread
              //KA_TRACE(15, ("__kmp_execute_tasks_template: T#%d NUMA stealing first from thread T#%d\n", gtid, tmp_victim_gtid));
              // fprintf(stderr, "NUMA steal: T#%d NUMA stealing first from thread T#%d\n", gtid, tmp_victim_gtid);
            }
            numa_thread = __kmp_threads[tmp_victim_gtid]; //threads_data[tmp_victim_tid].td.td_thr;
#if KMP_TASK_AFFINITY_MEASURE_TIME
            time1 = get_wall_time2()-time1;
            time2 = get_wall_time2();
#endif
            task = __kmp_steal_task(numa_thread, gtid, task_team,
                                    unfinished_threads, thread_finished,
                                    is_constrained);
#if KMP_TASK_AFFINITY_MEASURE_TIME
            time2 = get_wall_time2()-time2;
#endif
          }
          if(task != NULL) {
// #if KMP_TASK_AFFINITY_MEASURE_TIME
//             // stealing worked - so no overhead for kmp_steal_task counted
//             thread->th.th_sum_time_overhead_numa_task_stealing += time1;
//             thread->th.th_num_overhead_numa_task_stealing++;
// #endif

            victim =  __kmp_tid_from_gtid(tmp_victim_gtid);
            last_stolen_from_was_numa_thread = true;
            last_stolen_numa_thread = tmp_victim_gtid;

            // fprintf(stderr, "__kmp_execute_tasks_template: T#%d NUMA aware stealing worked .. reset counter\n", gtid);
            count_numa_tries = 0;

            // fprintf(stderr, "NUMA steal: T#%d NUMA stealing SUCCESS from thread T#%d\n", gtid, tmp_victim_gtid);
            //KA_TRACE(15, ("__kmp_execute_tasks_template: T#%d NUMA stealing SUCCESS from thread T#%d\n", gtid, tmp_victim_gtid));
          } else {
#if KMP_TASK_AFFINITY_MEASURE_TIME
            // numa stealing didnt work - so overhead for kmp_steal_task due to second call that is necessary
            // thread->th.th_sum_time_overhead_numa_task_stealing += time1+time2;
            thread->th.th_sum_time_overhead_numa_task_stealing += time2;
            thread->th.th_num_overhead_numa_task_stealing++;
#endif
            last_stolen_from_was_numa_thread = false;
            last_stolen_numa_thread = -1;

            // just do that if numa aware stealing is enabled
            //if(enable_numa_aware_stealing && task_team->tt.tt_numa_domains_set && *num_aff_tasks > 0) {
            if(enable_numa_aware_stealing && task_team->tt.tt_numa_domains_set) {
              count_numa_tries++;
              if(count_numa_tries <= max_numa_tries_before_normal){
                // stop here and retry numa aware stealing
                // fprintf(stderr, "__kmp_execute_tasks_template: T#%d NUMA aware stealing with max_numa_tries_before_normal = %d, count_numa_tries = %d\n", gtid, max_numa_tries_before_normal, count_numa_tries);
                continue;
              } else {
                // reset counter because maximum reached
                // fprintf(stderr, "__kmp_execute_tasks_template: T#%d NUMA aware stealing failed .. fallback stealing .. reset counter\n", gtid);
                count_numa_tries = 0;
              }
            }
            // fprintf(stderr, "NUMA steal: T#%d NUMA stealing noc successful from T#%d\n", gtid, tmp_victim_gtid);
#endif
          // We have a victim to try to steal from
          task = __kmp_steal_task(other_thread, gtid, task_team,
                                  unfinished_threads, thread_finished,
                                  is_constrained);
#if KMP_USE_TASK_AFFINITY && KMP_TASK_AFFINITY_NUMA_STEALING_ENABLED
          }
#endif
        }
        if (task != NULL) { // set last stolen to victim
          if (threads_data[tid].td.td_deque_last_stolen != victim) {
            threads_data[tid].td.td_deque_last_stolen = victim;
            // The pre-refactored code did not try more than 1 successful new
            // vicitm, unless the last one generated more local tasks;
            // new_victim keeps track of this
            new_victim = 1;
          }
        } else { // No tasks found; unset last_stolen
          KMP_CHECK_UPDATE(threads_data[tid].td.td_deque_last_stolen, -1);
          victim = -2; // no successful victim found
        }
      }

      if (task == NULL) // break out of tasking loop
        break;

// Found a task; execute it
#if USE_ITT_BUILD && USE_ITT_NOTIFY
      if (__itt_sync_create_ptr || KMP_ITT_DEBUG) {
        if (itt_sync_obj == NULL) { // we are at fork barrier where we could not
          // get the object reliably
          itt_sync_obj = __kmp_itt_barrier_object(gtid, bs_forkjoin_barrier);
        }
        __kmp_itt_task_starting(itt_sync_obj);
      }
#endif /* USE_ITT_BUILD && USE_ITT_NOTIFY */
// #if KMP_USE_TASK_AFFINITY && KMP_TASK_AFFINITY_PRINT_TASK_SIZE_EVOLUTION
//   // print information about nr of task currently assigned to this thread
//   double cur_time = get_wall_time2();
//   for(int i = 0; i < nthreads; i++)
//   {
//     int num_tasks = threads_data[i].td.td_deque_ntasks;
//     int cur_gtid = __kmp_gtid_from_thread(threads_data[i].td.td_thr);
//     fprintf(stderr, "__kmp_execute_tasks_template(task_aff_stats):\tT#%d\t%f\t%d\n", cur_gtid, cur_time, num_tasks);
//   }
// #endif
      __kmp_invoke_task(gtid, task, current_task);
#if USE_ITT_BUILD
      if (itt_sync_obj != NULL)
        __kmp_itt_task_finished(itt_sync_obj);
#endif /* USE_ITT_BUILD */
      // If this thread is only partway through the barrier and the condition is
      // met, then return now, so that the barrier gather/release pattern can
      // proceed. If this thread is in the last spin loop in the barrier,
      // waiting to be released, we know that the termination condition will not
      // be satisified, so don't waste any cycles checking it.
      if (flag == NULL || (!final_spin && flag->done_check())) {
        KA_TRACE(
            15,
            ("__kmp_execute_tasks_template: T#%d spin condition satisfied\n",
             gtid));
        return TRUE;
      }
      if (thread->th.th_task_team == NULL) {
        break;
      }
      // Yield before executing next task
      KMP_YIELD(__kmp_library == library_throughput);
      // If execution of a stolen task results in more tasks being placed on our
      // run queue, reset use_own_tasks
      if (!use_own_tasks && TCR_4(threads_data[tid].td.td_deque_ntasks) != 0) {
        KA_TRACE(20, ("__kmp_execute_tasks_template: T#%d stolen task spawned "
                      "other tasks, restart\n",
                      gtid));
        use_own_tasks = 1;
        new_victim = 0;
      }
    }

// The task source has been exhausted. If in final spin loop of barrier, check
// if termination condition is satisfied.
#if OMP_45_ENABLED
    // The work queue may be empty but there might be proxy tasks still
    // executing
    if (final_spin && TCR_4(current_task->td_incomplete_child_tasks) == 0)
#else
    if (final_spin)
#endif
    {
      // First, decrement the #unfinished threads, if that has not already been
      // done.  This decrement might be to the spin location, and result in the
      // termination condition being satisfied.
      if (!*thread_finished) {
        kmp_int32 count;

        count = KMP_TEST_THEN_DEC32(unfinished_threads) - 1; // TODO: Data Race????
        KA_TRACE(20, ("__kmp_execute_tasks_template: T#%d dec "
                      "unfinished_threads to %d task_team=%p\n",
                      gtid, count, task_team));
        *thread_finished = TRUE;
      }

      // It is now unsafe to reference thread->th.th_team !!!
      // Decrementing task_team->tt.tt_unfinished_threads can allow the master
      // thread to pass through the barrier, where it might reset each thread's
      // th.th_team field for the next parallel region. If we can steal more
      // work, we know that this has not happened yet.
      if (flag != NULL && flag->done_check()) {
        KA_TRACE(
            15,
            ("__kmp_execute_tasks_template: T#%d spin condition satisfied\n",
             gtid));
        return TRUE;
      }
    }

    // If this thread's task team is NULL, master has recognized that there are
    // no more tasks; bail out
    if (thread->th.th_task_team == NULL) {
      KA_TRACE(15,
               ("__kmp_execute_tasks_template: T#%d no more tasks\n", gtid));
      return FALSE;
    }

#if OMP_45_ENABLED
    // We could be getting tasks from target constructs; if this is the only
    // thread, keep trying to execute tasks from own queue
    if (nthreads == 1)
      use_own_tasks = 1;
    else
#endif
    {
      KA_TRACE(15,
               ("__kmp_execute_tasks_template: T#%d can't find work\n", gtid));
      return FALSE;
    }
  }
}

int __kmp_execute_tasks_32(
    kmp_info_t *thread, kmp_int32 gtid, kmp_flag_32 *flag, int final_spin,
    int *thread_finished USE_ITT_BUILD_ARG(void *itt_sync_obj),
    kmp_int32 is_constrained) {
  return __kmp_execute_tasks_template(
      thread, gtid, flag, final_spin,
      thread_finished USE_ITT_BUILD_ARG(itt_sync_obj), is_constrained);
}

int __kmp_execute_tasks_64(
    kmp_info_t *thread, kmp_int32 gtid, kmp_flag_64 *flag, int final_spin,
    int *thread_finished USE_ITT_BUILD_ARG(void *itt_sync_obj),
    kmp_int32 is_constrained) {
  return __kmp_execute_tasks_template(
      thread, gtid, flag, final_spin,
      thread_finished USE_ITT_BUILD_ARG(itt_sync_obj), is_constrained);
}

int __kmp_execute_tasks_oncore(
    kmp_info_t *thread, kmp_int32 gtid, kmp_flag_oncore *flag, int final_spin,
    int *thread_finished USE_ITT_BUILD_ARG(void *itt_sync_obj),
    kmp_int32 is_constrained) {
  return __kmp_execute_tasks_template(
      thread, gtid, flag, final_spin,
      thread_finished USE_ITT_BUILD_ARG(itt_sync_obj), is_constrained);
}

// __kmp_enable_tasking: Allocate task team and resume threads sleeping at the
// next barrier so they can assist in executing enqueued tasks.
// First thread in allocates the task team atomically.
static void __kmp_enable_tasking(kmp_task_team_t *task_team,
                                 kmp_info_t *this_thr) {
  kmp_thread_data_t *threads_data;
  int nthreads, i, is_init_thread;

  KA_TRACE(10, ("__kmp_enable_tasking(enter): T#%d\n",
                __kmp_gtid_from_thread(this_thr)));

  KMP_DEBUG_ASSERT(task_team != NULL);
  KMP_DEBUG_ASSERT(this_thr->th.th_team != NULL);

  nthreads = task_team->tt.tt_nproc;
  KMP_DEBUG_ASSERT(nthreads > 0);
  KMP_DEBUG_ASSERT(nthreads == this_thr->th.th_team->t.t_nproc);

  // Allocate or increase the size of threads_data if necessary
  is_init_thread = __kmp_realloc_task_threads_data(this_thr, task_team);

  if (!is_init_thread) {
    // Some other thread already set up the array.
    KA_TRACE(
        20,
        ("__kmp_enable_tasking(exit): T#%d: threads array already set up.\n",
         __kmp_gtid_from_thread(this_thr)));
    return;
  }
  threads_data = (kmp_thread_data_t *)TCR_PTR(task_team->tt.tt_threads_data);
  KMP_DEBUG_ASSERT(threads_data != NULL);

  if ((__kmp_tasking_mode == tskm_task_teams) &&
      (__kmp_dflt_blocktime != KMP_MAX_BLOCKTIME)) {
    // Release any threads sleeping at the barrier, so that they can steal
    // tasks and execute them.  In extra barrier mode, tasks do not sleep
    // at the separate tasking barrier, so this isn't a problem.
    for (i = 0; i < nthreads; i++) {
      volatile void *sleep_loc;
      kmp_info_t *thread = threads_data[i].td.td_thr;

      if (i == this_thr->th.th_info.ds.ds_tid) {
        continue;
      }
      // Since we haven't locked the thread's suspend mutex lock at this
      // point, there is a small window where a thread might be putting
      // itself to sleep, but hasn't set the th_sleep_loc field yet.
      // To work around this, __kmp_execute_tasks_template() periodically checks
      // see if other threads are sleeping (using the same random mechanism that
      // is used for task stealing) and awakens them if they are.
      if ((sleep_loc = TCR_PTR(CCAST(void *, thread->th.th_sleep_loc))) !=
          NULL) {
        KF_TRACE(50, ("__kmp_enable_tasking: T#%d waking up thread T#%d\n",
                      __kmp_gtid_from_thread(this_thr),
                      __kmp_gtid_from_thread(thread)));
        __kmp_null_resume_wrapper(__kmp_gtid_from_thread(thread), sleep_loc);
      } else {
        KF_TRACE(50, ("__kmp_enable_tasking: T#%d don't wake up thread T#%d\n",
                      __kmp_gtid_from_thread(this_thr),
                      __kmp_gtid_from_thread(thread)));
      }
    }
  }

  KA_TRACE(10, ("__kmp_enable_tasking(exit): T#%d\n",
                __kmp_gtid_from_thread(this_thr)));
}

/* // TODO: Check the comment consistency
 * Utility routines for "task teams".  A task team (kmp_task_t) is kind of
 * like a shadow of the kmp_team_t data struct, with a different lifetime.
 * After a child * thread checks into a barrier and calls __kmp_release() from
 * the particular variant of __kmp_<barrier_kind>_barrier_gather(), it can no
 * longer assume that the kmp_team_t structure is intact (at any moment, the
 * master thread may exit the barrier code and free the team data structure,
 * and return the threads to the thread pool).
 *
 * This does not work with the the tasking code, as the thread is still
 * expected to participate in the execution of any tasks that may have been
 * spawned my a member of the team, and the thread still needs access to all
 * to each thread in the team, so that it can steal work from it.
 *
 * Enter the existence of the kmp_task_team_t struct.  It employs a reference
 * counting mechanims, and is allocated by the master thread before calling
 * __kmp_<barrier_kind>_release, and then is release by the last thread to
 * exit __kmp_<barrier_kind>_release at the next barrier.  I.e. the lifetimes
 * of the kmp_task_team_t structs for consecutive barriers can overlap
 * (and will, unless the master thread is the last thread to exit the barrier
 * release phase, which is not typical).
 *
 * The existence of such a struct is useful outside the context of tasking,
 * but for now, I'm trying to keep it specific to the OMP_30_ENABLED macro,
 * so that any performance differences show up when comparing the 2.5 vs. 3.0
 * libraries.
 *
 * We currently use the existence of the threads array as an indicator that
 * tasks were spawned since the last barrier.  If the structure is to be
 * useful outside the context of tasking, then this will have to change, but
 * not settting the field minimizes the performance impact of tasking on
 * barriers, when no explicit tasks were spawned (pushed, actually).
 */

static kmp_task_team_t *__kmp_free_task_teams =
    NULL; // Free list for task_team data structures
// Lock for task team data structures
static kmp_bootstrap_lock_t __kmp_task_team_lock =
    KMP_BOOTSTRAP_LOCK_INITIALIZER(__kmp_task_team_lock);

// __kmp_alloc_task_deque:
// Allocates a task deque for a particular thread, and initialize the necessary
// data structures relating to the deque.  This only happens once per thread
// per task team since task teams are recycled. No lock is needed during
// allocation since each thread allocates its own deque.
static void __kmp_alloc_task_deque(kmp_info_t *thread,
                                   kmp_thread_data_t *thread_data) {
  __kmp_init_bootstrap_lock(&thread_data->td.td_deque_lock);
  KMP_DEBUG_ASSERT(thread_data->td.td_deque == NULL);

  // Initialize last stolen task field to "none"
  thread_data->td.td_deque_last_stolen = -1;

  KMP_DEBUG_ASSERT(TCR_4(thread_data->td.td_deque_ntasks) == 0);
  KMP_DEBUG_ASSERT(thread_data->td.td_deque_head == 0);
  KMP_DEBUG_ASSERT(thread_data->td.td_deque_tail == 0);

  KA_TRACE(
      10,
      ("__kmp_alloc_task_deque: T#%d allocating deque[%d] for thread_data %p\n",
       __kmp_gtid_from_thread(thread), INITIAL_TASK_DEQUE_SIZE, thread_data));
  // Allocate space for task deque, and zero the deque
  // Cannot use __kmp_thread_calloc() because threads not around for
  // kmp_reap_task_team( ).
  thread_data->td.td_deque = (kmp_taskdata_t **)__kmp_allocate(
      INITIAL_TASK_DEQUE_SIZE * sizeof(kmp_taskdata_t *));
  thread_data->td.td_deque_size = INITIAL_TASK_DEQUE_SIZE;
}

// __kmp_realloc_task_deque:
// Re-allocates a task deque for a particular thread, copies the content from
// the old deque and adjusts the necessary data structures relating to the
// deque. This operation must be done with a the deque_lock being held
static void __kmp_realloc_task_deque(kmp_info_t *thread,
                                     kmp_thread_data_t *thread_data) {
  kmp_int32 size = TASK_DEQUE_SIZE(thread_data->td);
  kmp_int32 new_size = 2 * size;

  KA_TRACE(10, ("__kmp_realloc_task_deque: T#%d reallocating deque[from %d to "
                "%d] for thread_data %p\n",
                __kmp_gtid_from_thread(thread), size, new_size, thread_data));

  kmp_taskdata_t **new_deque =
      (kmp_taskdata_t **)__kmp_allocate(new_size * sizeof(kmp_taskdata_t *));

  int i, j;
  for (i = thread_data->td.td_deque_head, j = 0; j < size;
       i = (i + 1) & TASK_DEQUE_MASK(thread_data->td), j++)
    new_deque[j] = thread_data->td.td_deque[i];

  __kmp_free(thread_data->td.td_deque);

  thread_data->td.td_deque_head = 0;
  thread_data->td.td_deque_tail = size;
  thread_data->td.td_deque = new_deque;
  thread_data->td.td_deque_size = new_size;
}

// __kmp_free_task_deque:
// Deallocates a task deque for a particular thread. Happens at library
// deallocation so don't need to reset all thread data fields.
static void __kmp_free_task_deque(kmp_thread_data_t *thread_data) {
  if (thread_data->td.td_deque != NULL) {
    __kmp_acquire_bootstrap_lock(&thread_data->td.td_deque_lock);
    TCW_4(thread_data->td.td_deque_ntasks, 0);
    __kmp_free(thread_data->td.td_deque);
    thread_data->td.td_deque = NULL;
    __kmp_release_bootstrap_lock(&thread_data->td.td_deque_lock);
  }

#ifdef BUILD_TIED_TASK_STACK
  // GEH: Figure out what to do here for td_susp_tied_tasks
  if (thread_data->td.td_susp_tied_tasks.ts_entries != TASK_STACK_EMPTY) {
    __kmp_free_task_stack(__kmp_thread_from_gtid(gtid), thread_data);
  }
#endif // BUILD_TIED_TASK_STACK
}

// __kmp_realloc_task_threads_data:
// Allocates a threads_data array for a task team, either by allocating an
// initial array or enlarging an existing array.  Only the first thread to get
// the lock allocs or enlarges the array and re-initializes the array eleemnts.
// That thread returns "TRUE", the rest return "FALSE".
// Assumes that the new array size is given by task_team -> tt.tt_nproc.
// The current size is given by task_team -> tt.tt_max_threads.
static int __kmp_realloc_task_threads_data(kmp_info_t *thread,
                                           kmp_task_team_t *task_team) {
  kmp_thread_data_t **threads_data_p;
  kmp_int32 nthreads, maxthreads;
  int is_init_thread = FALSE;

  if (TCR_4(task_team->tt.tt_found_tasks)) {
    // Already reallocated and initialized.
    return FALSE;
  }

  threads_data_p = &task_team->tt.tt_threads_data;
  nthreads = task_team->tt.tt_nproc;
  maxthreads = task_team->tt.tt_max_threads;

  // All threads must lock when they encounter the first task of the implicit
  // task region to make sure threads_data fields are (re)initialized before
  // used.
  __kmp_acquire_bootstrap_lock(&task_team->tt.tt_threads_lock);

  if (!TCR_4(task_team->tt.tt_found_tasks)) {
    // first thread to enable tasking
    kmp_team_t *team = thread->th.th_team;
    int i;

    is_init_thread = TRUE;
    if (maxthreads < nthreads) {

      if (*threads_data_p != NULL) {
        kmp_thread_data_t *old_data = *threads_data_p;
        kmp_thread_data_t *new_data = NULL;

        KA_TRACE(
            10,
            ("__kmp_realloc_task_threads_data: T#%d reallocating "
             "threads data for task_team %p, new_size = %d, old_size = %d\n",
             __kmp_gtid_from_thread(thread), task_team, nthreads, maxthreads));
        // Reallocate threads_data to have more elements than current array
        // Cannot use __kmp_thread_realloc() because threads not around for
        // kmp_reap_task_team( ).  Note all new array entries are initialized
        // to zero by __kmp_allocate().
        new_data = (kmp_thread_data_t *)__kmp_allocate(
            nthreads * sizeof(kmp_thread_data_t));
        // copy old data to new data
        KMP_MEMCPY_S((void *)new_data, nthreads * sizeof(kmp_thread_data_t),
                     (void *)old_data, maxthreads * sizeof(kmp_thread_data_t));

#ifdef BUILD_TIED_TASK_STACK
        // GEH: Figure out if this is the right thing to do
        for (i = maxthreads; i < nthreads; i++) {
          kmp_thread_data_t *thread_data = &(*threads_data_p)[i];
          __kmp_init_task_stack(__kmp_gtid_from_thread(thread), thread_data);
        }
#endif // BUILD_TIED_TASK_STACK
        // Install the new data and free the old data
        (*threads_data_p) = new_data;
        __kmp_free(old_data);
      } else {
        KA_TRACE(10, ("__kmp_realloc_task_threads_data: T#%d allocating "
                      "threads data for task_team %p, size = %d\n",
                      __kmp_gtid_from_thread(thread), task_team, nthreads));
        // Make the initial allocate for threads_data array, and zero entries
        // Cannot use __kmp_thread_calloc() because threads not around for
        // kmp_reap_task_team( ).
        ANNOTATE_IGNORE_WRITES_BEGIN();
        *threads_data_p = (kmp_thread_data_t *)__kmp_allocate(
            nthreads * sizeof(kmp_thread_data_t));
        ANNOTATE_IGNORE_WRITES_END();
#ifdef BUILD_TIED_TASK_STACK
        // GEH: Figure out if this is the right thing to do
        for (i = 0; i < nthreads; i++) {
          kmp_thread_data_t *thread_data = &(*threads_data_p)[i];
          __kmp_init_task_stack(__kmp_gtid_from_thread(thread), thread_data);
        }
#endif // BUILD_TIED_TASK_STACK
      }
      task_team->tt.tt_max_threads = nthreads;
    } else {
      // If array has (more than) enough elements, go ahead and use it
      KMP_DEBUG_ASSERT(*threads_data_p != NULL);
    }

    // initialize threads_data pointers back to thread_info structures
    for (i = 0; i < nthreads; i++) {
      kmp_thread_data_t *thread_data = &(*threads_data_p)[i];
      thread_data->td.td_thr = team->t.t_threads[i];

      if (thread_data->td.td_deque_last_stolen >= nthreads) {
        // The last stolen field survives across teams / barrier, and the number
        // of threads may have changed.  It's possible (likely?) that a new
        // parallel region will exhibit the same behavior as previous region.
        thread_data->td.td_deque_last_stolen = -1;
      }

#if KMP_USE_TASK_AFFINITY
      thread_data->td.td_idx_in_numa_map = -1;
#endif
    }

    KMP_MB();
    TCW_SYNC_4(task_team->tt.tt_found_tasks, TRUE);
  }

  __kmp_release_bootstrap_lock(&task_team->tt.tt_threads_lock);
  return is_init_thread;
}

// __kmp_free_task_threads_data:
// Deallocates a threads_data array for a task team, including any attached
// tasking deques.  Only occurs at library shutdown.
static void __kmp_free_task_threads_data(kmp_task_team_t *task_team) {
  __kmp_acquire_bootstrap_lock(&task_team->tt.tt_threads_lock);
  if (task_team->tt.tt_threads_data != NULL) {
    int i;
    for (i = 0; i < task_team->tt.tt_max_threads; i++) {
      __kmp_free_task_deque(&task_team->tt.tt_threads_data[i]);
    }
    __kmp_free(task_team->tt.tt_threads_data);
    task_team->tt.tt_threads_data = NULL;
  }
  __kmp_release_bootstrap_lock(&task_team->tt.tt_threads_lock);
}

// __kmp_allocate_task_team:
// Allocates a task team associated with a specific team, taking it from
// the global task team free list if possible.  Also initializes data
// structures.
static kmp_task_team_t *__kmp_allocate_task_team(kmp_info_t *thread,
                                                 kmp_team_t *team) {
  kmp_task_team_t *task_team = NULL;
  int nthreads;

  KA_TRACE(20, ("__kmp_allocate_task_team: T#%d entering; team = %p\n",
                (thread ? __kmp_gtid_from_thread(thread) : -1), team));

  if (TCR_PTR(__kmp_free_task_teams) != NULL) {
    // Take a task team from the task team pool
    __kmp_acquire_bootstrap_lock(&__kmp_task_team_lock);
    if (__kmp_free_task_teams != NULL) {
      task_team = __kmp_free_task_teams;
      TCW_PTR(__kmp_free_task_teams, task_team->tt.tt_next);
      task_team->tt.tt_next = NULL;
    }
    __kmp_release_bootstrap_lock(&__kmp_task_team_lock);
  }

  if (task_team == NULL) {
    KA_TRACE(10, ("__kmp_allocate_task_team: T#%d allocating "
                  "task team for team %p\n",
                  __kmp_gtid_from_thread(thread), team));
    // Allocate a new task team if one is not available.
    // Cannot use __kmp_thread_malloc() because threads not around for
    // kmp_reap_task_team( ).
    task_team = (kmp_task_team_t *)__kmp_allocate(sizeof(kmp_task_team_t));
    __kmp_init_bootstrap_lock(&task_team->tt.tt_threads_lock);
    // AC: __kmp_allocate zeroes returned memory
    // task_team -> tt.tt_threads_data = NULL;
    // task_team -> tt.tt_max_threads = 0;
    // task_team -> tt.tt_next = NULL;
  }

#if KMP_USE_TASK_AFFINITY
  TCW_4(task_team->tt.tt_numa_domains_set, FALSE);
  // init lock here
  __kmp_init_bootstrap_lock(&task_team->tt.tt_lock_numa_map);

  __kmp_init_bootstrap_lock(&task_team->tt.tt_lock_tasks_with_affinity);
  task_team->tt.tt_num_tasks_with_aff = 0;
#endif

  TCW_4(task_team->tt.tt_found_tasks, FALSE);
#if OMP_45_ENABLED
  TCW_4(task_team->tt.tt_found_proxy_tasks, FALSE);
#endif
  task_team->tt.tt_nproc = nthreads = team->t.t_nproc;

  TCW_4(task_team->tt.tt_unfinished_threads, nthreads);
  TCW_4(task_team->tt.tt_active, TRUE);

  KA_TRACE(20, ("__kmp_allocate_task_team: T#%d exiting; task_team = %p "
                "unfinished_threads init'd to %d\n",
                (thread ? __kmp_gtid_from_thread(thread) : -1), task_team,
                task_team->tt.tt_unfinished_threads));
  return task_team;
}

// __kmp_free_task_team:
// Frees the task team associated with a specific thread, and adds it
// to the global task team free list.
void __kmp_free_task_team(kmp_info_t *thread, kmp_task_team_t *task_team) {
  KA_TRACE(20, ("__kmp_free_task_team: T#%d task_team = %p\n",
                thread ? __kmp_gtid_from_thread(thread) : -1, task_team));

  // Put task team back on free list
  __kmp_acquire_bootstrap_lock(&__kmp_task_team_lock);

#if KMP_USE_TASK_AFFINITY && KMP_TASK_AFFINITY_NEW_MEMORY_ALLOC
  int i;
  for (i = 0; i < task_team->tt.tt_num_numa_domains; i++){
    free(task_team->tt.tt_map_threads_in_domain[i]);
  }
  free(task_team->tt.tt_numa_domain_size);
  free(task_team->tt.tt_numa_domain_rr_counter);
  free(task_team->tt.tt_map_threads_in_domain);
#endif

  KMP_DEBUG_ASSERT(task_team->tt.tt_next == NULL);
  task_team->tt.tt_next = __kmp_free_task_teams;
  TCW_PTR(__kmp_free_task_teams, task_team);

  __kmp_release_bootstrap_lock(&__kmp_task_team_lock);
}

// __kmp_reap_task_teams:
// Free all the task teams on the task team free list.
// Should only be done during library shutdown.
// Cannot do anything that needs a thread structure or gtid since they are
// already gone.
void __kmp_reap_task_teams(void) {
  kmp_task_team_t *task_team;

  if (TCR_PTR(__kmp_free_task_teams) != NULL) {
    // Free all task_teams on the free list
    __kmp_acquire_bootstrap_lock(&__kmp_task_team_lock);
    while ((task_team = __kmp_free_task_teams) != NULL) {
      __kmp_free_task_teams = task_team->tt.tt_next;
      task_team->tt.tt_next = NULL;

      // Free threads_data if necessary
      if (task_team->tt.tt_threads_data != NULL) {
        __kmp_free_task_threads_data(task_team);
      }
      __kmp_free(task_team);
    }
    __kmp_release_bootstrap_lock(&__kmp_task_team_lock);
  }
}

// __kmp_wait_to_unref_task_teams:
// Some threads could still be in the fork barrier release code, possibly
// trying to steal tasks.  Wait for each thread to unreference its task team.
void __kmp_wait_to_unref_task_teams(void) {
  kmp_info_t *thread;
  kmp_uint32 spins;
  int done;

  KMP_INIT_YIELD(spins);

  for (;;) {
    done = TRUE;

    // TODO: GEH - this may be is wrong because some sync would be necessary
    // in case threads are added to the pool during the traversal. Need to
    // verify that lock for thread pool is held when calling this routine.
    for (thread = CCAST(kmp_info_t *, __kmp_thread_pool); thread != NULL;
         thread = thread->th.th_next_pool) {
#if KMP_OS_WINDOWS
      DWORD exit_val;
#endif
      if (TCR_PTR(thread->th.th_task_team) == NULL) {
        KA_TRACE(10, ("__kmp_wait_to_unref_task_team: T#%d task_team == NULL\n",
                      __kmp_gtid_from_thread(thread)));
        continue;
      }
#if KMP_OS_WINDOWS
      // TODO: GEH - add this check for Linux* OS / OS X* as well?
      if (!__kmp_is_thread_alive(thread, &exit_val)) {
        thread->th.th_task_team = NULL;
        continue;
      }
#endif

      done = FALSE; // Because th_task_team pointer is not NULL for this thread

      KA_TRACE(10, ("__kmp_wait_to_unref_task_team: Waiting for T#%d to "
                    "unreference task_team\n",
                    __kmp_gtid_from_thread(thread)));

      if (__kmp_dflt_blocktime != KMP_MAX_BLOCKTIME) {
        volatile void *sleep_loc;
        // If the thread is sleeping, awaken it.
        if ((sleep_loc = TCR_PTR(CCAST(void *, thread->th.th_sleep_loc))) !=
            NULL) {
          KA_TRACE(
              10,
              ("__kmp_wait_to_unref_task_team: T#%d waking up thread T#%d\n",
               __kmp_gtid_from_thread(thread), __kmp_gtid_from_thread(thread)));
          __kmp_null_resume_wrapper(__kmp_gtid_from_thread(thread), sleep_loc);
        }
      }
    }
    if (done) {
      break;
    }

    // If we are oversubscribed, or have waited a bit (and library mode is
    // throughput), yield. Pause is in the following code.
    KMP_YIELD(TCR_4(__kmp_nth) > __kmp_avail_proc);
    KMP_YIELD_SPIN(spins); // Yields only if KMP_LIBRARY=throughput
  }
}

// __kmp_task_team_setup:  Create a task_team for the current team, but use
// an already created, unused one if it already exists.
void __kmp_task_team_setup(kmp_info_t *this_thr, kmp_team_t *team, int always) {
  KMP_DEBUG_ASSERT(__kmp_tasking_mode != tskm_immediate_exec);

  // If this task_team hasn't been created yet, allocate it. It will be used in
  // the region after the next.
  // If it exists, it is the current task team and shouldn't be touched yet as
  // it may still be in use.
  if (team->t.t_task_team[this_thr->th.th_task_state] == NULL &&
      (always || team->t.t_nproc > 1)) {
    team->t.t_task_team[this_thr->th.th_task_state] =
        __kmp_allocate_task_team(this_thr, team);
    KA_TRACE(20, ("__kmp_task_team_setup: Master T#%d created new task_team %p "
                  "for team %d at parity=%d\n",
                  __kmp_gtid_from_thread(this_thr),
                  team->t.t_task_team[this_thr->th.th_task_state],
                  ((team != NULL) ? team->t.t_id : -1),
                  this_thr->th.th_task_state));
  }

  // After threads exit the release, they will call sync, and then point to this
  // other task_team; make sure it is allocated and properly initialized. As
  // threads spin in the barrier release phase, they will continue to use the
  // previous task_team struct(above), until they receive the signal to stop
  // checking for tasks (they can't safely reference the kmp_team_t struct,
  // which could be reallocated by the master thread). No task teams are formed
  // for serialized teams.
  if (team->t.t_nproc > 1) {
    int other_team = 1 - this_thr->th.th_task_state;
    if (team->t.t_task_team[other_team] == NULL) { // setup other team as well
      team->t.t_task_team[other_team] =
          __kmp_allocate_task_team(this_thr, team);
      KA_TRACE(20, ("__kmp_task_team_setup: Master T#%d created second new "
                    "task_team %p for team %d at parity=%d\n",
                    __kmp_gtid_from_thread(this_thr),
                    team->t.t_task_team[other_team],
                    ((team != NULL) ? team->t.t_id : -1), other_team));
    } else { // Leave the old task team struct in place for the upcoming region;
      // adjust as needed
      kmp_task_team_t *task_team = team->t.t_task_team[other_team];
      if (!task_team->tt.tt_active ||
          team->t.t_nproc != task_team->tt.tt_nproc) {
        TCW_4(task_team->tt.tt_nproc, team->t.t_nproc);
        TCW_4(task_team->tt.tt_found_tasks, FALSE);
#if OMP_45_ENABLED
        TCW_4(task_team->tt.tt_found_proxy_tasks, FALSE);
#endif
        TCW_4(task_team->tt.tt_unfinished_threads, team->t.t_nproc);
        TCW_4(task_team->tt.tt_active, TRUE);
      }
      // if team size has changed, the first thread to enable tasking will
      // realloc threads_data if necessary
      KA_TRACE(20, ("__kmp_task_team_setup: Master T#%d reset next task_team "
                    "%p for team %d at parity=%d\n",
                    __kmp_gtid_from_thread(this_thr),
                    team->t.t_task_team[other_team],
                    ((team != NULL) ? team->t.t_id : -1), other_team));
    }
  }
}

// __kmp_task_team_sync: Propagation of task team data from team to threads
// which happens just after the release phase of a team barrier.  This may be
// called by any thread, but only for teams with # threads > 1.
void __kmp_task_team_sync(kmp_info_t *this_thr, kmp_team_t *team) {
  KMP_DEBUG_ASSERT(__kmp_tasking_mode != tskm_immediate_exec);

  // Toggle the th_task_state field, to switch which task_team this thread
  // refers to
  this_thr->th.th_task_state = 1 - this_thr->th.th_task_state;
  // It is now safe to propagate the task team pointer from the team struct to
  // the current thread.
  TCW_PTR(this_thr->th.th_task_team,
          team->t.t_task_team[this_thr->th.th_task_state]);
  KA_TRACE(20,
           ("__kmp_task_team_sync: Thread T#%d task team switched to task_team "
            "%p from Team #%d (parity=%d)\n",
            __kmp_gtid_from_thread(this_thr), this_thr->th.th_task_team,
            ((team != NULL) ? team->t.t_id : -1), this_thr->th.th_task_state));
}

// __kmp_task_team_wait: Master thread waits for outstanding tasks after the
// barrier gather phase. Only called by master thread if #threads in team > 1 or
// if proxy tasks were created.
//
// wait is a flag that defaults to 1 (see kmp.h), but waiting can be turned off
// by passing in 0 optionally as the last argument. When wait is zero, master
// thread does not wait for unfinished_threads to reach 0.
void __kmp_task_team_wait(
    kmp_info_t *this_thr,
    kmp_team_t *team USE_ITT_BUILD_ARG(void *itt_sync_obj), int wait) {
  kmp_task_team_t *task_team = team->t.t_task_team[this_thr->th.th_task_state];

  KMP_DEBUG_ASSERT(__kmp_tasking_mode != tskm_immediate_exec);
  KMP_DEBUG_ASSERT(task_team == this_thr->th.th_task_team);

  if ((task_team != NULL) && KMP_TASKING_ENABLED(task_team)) {
    if (wait) {
      KA_TRACE(20, ("__kmp_task_team_wait: Master T#%d waiting for all tasks "
                    "(for unfinished_threads to reach 0) on task_team = %p\n",
                    __kmp_gtid_from_thread(this_thr), task_team));
      // Worker threads may have dropped through to release phase, but could
      // still be executing tasks. Wait here for tasks to complete. To avoid
      // memory contention, only master thread checks termination condition.
      kmp_flag_32 flag(
          RCAST(volatile kmp_uint32 *, &task_team->tt.tt_unfinished_threads),
          0U);
      flag.wait(this_thr, TRUE USE_ITT_BUILD_ARG(itt_sync_obj));
    }
    // Deactivate the old task team, so that the worker threads will stop
    // referencing it while spinning.
    KA_TRACE(
        20,
        ("__kmp_task_team_wait: Master T#%d deactivating task_team %p: "
         "setting active to false, setting local and team's pointer to NULL\n",
         __kmp_gtid_from_thread(this_thr), task_team));
#if OMP_45_ENABLED
    KMP_DEBUG_ASSERT(task_team->tt.tt_nproc > 1 ||
                     task_team->tt.tt_found_proxy_tasks == TRUE);
    TCW_SYNC_4(task_team->tt.tt_found_proxy_tasks, FALSE);
#else
    KMP_DEBUG_ASSERT(task_team->tt.tt_nproc > 1);
#endif
    TCW_SYNC_4(task_team->tt.tt_active, FALSE);
    KMP_MB();

    TCW_PTR(this_thr->th.th_task_team, NULL);
  }
}

// __kmp_tasking_barrier:
// This routine may only called when __kmp_tasking_mode == tskm_extra_barrier.
// Internal function to execute all tasks prior to a regular barrier or a join
// barrier. It is a full barrier itself, which unfortunately turns regular
// barriers into double barriers and join barriers into 1 1/2 barriers.
void __kmp_tasking_barrier(kmp_team_t *team, kmp_info_t *thread, int gtid) {
  volatile kmp_uint32 *spin = RCAST(
      volatile kmp_uint32 *,
      &team->t.t_task_team[thread->th.th_task_state]->tt.tt_unfinished_threads);
  int flag = FALSE;
  KMP_DEBUG_ASSERT(__kmp_tasking_mode == tskm_extra_barrier);

#if USE_ITT_BUILD
  KMP_FSYNC_SPIN_INIT(spin, (kmp_uint32 *)NULL);
#endif /* USE_ITT_BUILD */
  kmp_flag_32 spin_flag(spin, 0U);
  while (!spin_flag.execute_tasks(thread, gtid, TRUE,
                                  &flag USE_ITT_BUILD_ARG(NULL), 0)) {
#if USE_ITT_BUILD
    // TODO: What about itt_sync_obj??
    KMP_FSYNC_SPIN_PREPARE(CCAST(kmp_uint32 *, spin));
#endif /* USE_ITT_BUILD */

    if (TCR_4(__kmp_global.g.g_done)) {
      if (__kmp_global.g.g_abort)
        __kmp_abort_thread();
      break;
    }
    KMP_YIELD(TRUE); // GH: We always yield here
  }
#if USE_ITT_BUILD
  KMP_FSYNC_SPIN_ACQUIRED(CCAST(kmp_uint32 *, spin));
#endif /* USE_ITT_BUILD */
}

#if OMP_45_ENABLED

// __kmp_give_task puts a task into a given thread queue if:
//  - the queue for that thread was created
//  - there's space in that queue
// Because of this, __kmp_push_task needs to check if there's space after
// getting the lock
static bool __kmp_give_task(kmp_info_t *thread, kmp_int32 tid, kmp_task_t *task,
                            kmp_int32 pass) {
  kmp_taskdata_t *taskdata = KMP_TASK_TO_TASKDATA(task);
  kmp_task_team_t *task_team = taskdata->td_task_team;

  KA_TRACE(20, ("__kmp_give_task: T#%d trying to give task %p to thread %d.\n", __kmp_entry_gtid(),
                taskdata, tid));

  // If task_team is NULL something went really bad...
  KMP_DEBUG_ASSERT(task_team != NULL);

  bool result = false;
  kmp_thread_data_t *thread_data = &task_team->tt.tt_threads_data[tid];

  if (thread_data->td.td_deque == NULL) {
    // There's no queue in this thread, go find another one
    // We're guaranteed that at least one thread has a queue
    KA_TRACE(30,
             ("__kmp_give_task: thread %d has no queue while giving task %p.\n",
              tid, taskdata));
    return result;
  }

  if (TCR_4(thread_data->td.td_deque_ntasks) >=
      TASK_DEQUE_SIZE(thread_data->td)) {
    KA_TRACE(
        30,
        ("__kmp_give_task: queue is full while giving task %p to thread %d.\n",
         taskdata, tid));

    // if this deque is bigger than the pass ratio give a chance to another
    // thread
    if (TASK_DEQUE_SIZE(thread_data->td) / INITIAL_TASK_DEQUE_SIZE >= pass)
      return result;

    __kmp_acquire_bootstrap_lock(&thread_data->td.td_deque_lock);
    __kmp_realloc_task_deque(thread, thread_data);

  } else {

    __kmp_acquire_bootstrap_lock(&thread_data->td.td_deque_lock);

    if (TCR_4(thread_data->td.td_deque_ntasks) >=
        TASK_DEQUE_SIZE(thread_data->td)) {
      KA_TRACE(30, ("__kmp_give_task: queue is full while giving task %p to "
                    "thread %d.\n",
                    taskdata, tid));

      // if this deque is bigger than the pass ratio give a chance to another
      // thread
      if (TASK_DEQUE_SIZE(thread_data->td) / INITIAL_TASK_DEQUE_SIZE >= pass)
        goto release_and_exit;

      __kmp_realloc_task_deque(thread, thread_data);
    }
  }

  // lock is held here, and there is space in the deque

  thread_data->td.td_deque[thread_data->td.td_deque_tail] = taskdata;
  // Wrap index.
  thread_data->td.td_deque_tail =
      (thread_data->td.td_deque_tail + 1) & TASK_DEQUE_MASK(thread_data->td);
  TCW_4(thread_data->td.td_deque_ntasks,
        TCR_4(thread_data->td.td_deque_ntasks) + 1);

  result = true;
  KA_TRACE(30, ("__kmp_give_task: successfully gave task %p to thread %d.\n",
                taskdata, tid));

release_and_exit:
  __kmp_release_bootstrap_lock(&thread_data->td.td_deque_lock);

  return result;
}

/* The finish of the proxy tasks is divided in two pieces:
    - the top half is the one that can be done from a thread outside the team
    - the bottom half must be run from a them within the team

   In order to run the bottom half the task gets queued back into one of the
   threads of the team. Once the td_incomplete_child_task counter of the parent
   is decremented the threads can leave the barriers. So, the bottom half needs
   to be queued before the counter is decremented. The top half is therefore
   divided in two parts:
    - things that can be run before queuing the bottom half
    - things that must be run after queuing the bottom half

   This creates a second race as the bottom half can free the task before the
   second top half is executed. To avoid this we use the
   td_incomplete_child_task of the proxy task to synchronize the top and bottom
   half. */
static void __kmp_first_top_half_finish_proxy(kmp_taskdata_t *taskdata) {
  KMP_DEBUG_ASSERT(taskdata->td_flags.tasktype == TASK_EXPLICIT);
  KMP_DEBUG_ASSERT(taskdata->td_flags.proxy == TASK_PROXY);
  KMP_DEBUG_ASSERT(taskdata->td_flags.complete == 0);
  KMP_DEBUG_ASSERT(taskdata->td_flags.freed == 0);

  taskdata->td_flags.complete = 1; // mark the task as completed

  if (taskdata->td_taskgroup)
    KMP_TEST_THEN_DEC32(&taskdata->td_taskgroup->count);

  // Create an imaginary children for this task so the bottom half cannot
  // release the task before we have completed the second top half
  TCI_4(taskdata->td_incomplete_child_tasks);
}

static void __kmp_second_top_half_finish_proxy(kmp_taskdata_t *taskdata) {
  kmp_int32 children = 0;

  // Predecrement simulated by "- 1" calculation
  children =
      KMP_TEST_THEN_DEC32(&taskdata->td_parent->td_incomplete_child_tasks) - 1;
  KMP_DEBUG_ASSERT(children >= 0);

  // Remove the imaginary children
  TCD_4(taskdata->td_incomplete_child_tasks);
}

static void __kmp_bottom_half_finish_proxy(kmp_int32 gtid, kmp_task_t *ptask) {
  kmp_taskdata_t *taskdata = KMP_TASK_TO_TASKDATA(ptask);
  kmp_info_t *thread = __kmp_threads[gtid];

  KMP_DEBUG_ASSERT(taskdata->td_flags.proxy == TASK_PROXY);
  KMP_DEBUG_ASSERT(taskdata->td_flags.complete ==
                   1); // top half must run before bottom half

  // We need to wait to make sure the top half is finished
  // Spinning here should be ok as this should happen quickly
  while (TCR_4(taskdata->td_incomplete_child_tasks) > 0)
    ;

  __kmp_release_deps(gtid, taskdata);
  __kmp_free_task_and_ancestors(gtid, taskdata, thread);
}

/*!
@ingroup TASKING
@param gtid Global Thread ID of encountering thread
@param ptask Task which execution is completed

Execute the completation of a proxy task from a thread of that is part of the
team. Run first and bottom halves directly.
*/
void __kmpc_proxy_task_completed(kmp_int32 gtid, kmp_task_t *ptask) {
  KMP_DEBUG_ASSERT(ptask != NULL);
  kmp_taskdata_t *taskdata = KMP_TASK_TO_TASKDATA(ptask);
  KA_TRACE(
      10, ("__kmp_proxy_task_completed(enter): T#%d proxy task %p completing\n",
           gtid, taskdata));

  KMP_DEBUG_ASSERT(taskdata->td_flags.proxy == TASK_PROXY);

  __kmp_first_top_half_finish_proxy(taskdata);
  __kmp_second_top_half_finish_proxy(taskdata);
  __kmp_bottom_half_finish_proxy(gtid, ptask);

  KA_TRACE(10,
           ("__kmp_proxy_task_completed(exit): T#%d proxy task %p completing\n",
            gtid, taskdata));
}

/*!
@ingroup TASKING
@param ptask Task which execution is completed

Execute the completation of a proxy task from a thread that could not belong to
the team.
*/
void __kmpc_proxy_task_completed_ooo(kmp_task_t *ptask) {
  KMP_DEBUG_ASSERT(ptask != NULL);
  kmp_taskdata_t *taskdata = KMP_TASK_TO_TASKDATA(ptask);

  KA_TRACE(
      10,
      ("__kmp_proxy_task_completed_ooo(enter): proxy task completing ooo %p\n",
       taskdata));

  KMP_DEBUG_ASSERT(taskdata->td_flags.proxy == TASK_PROXY);

  __kmp_first_top_half_finish_proxy(taskdata);

  // Enqueue task to complete bottom half completion from a thread within the
  // corresponding team
  kmp_team_t *team = taskdata->td_team;
  kmp_int32 nthreads = team->t.t_nproc;
  kmp_info_t *thread;

  // This should be similar to start_k = __kmp_get_random( thread ) % nthreads
  // but we cannot use __kmp_get_random here
  kmp_int32 start_k = 0;
  kmp_int32 pass = 1;
  kmp_int32 k = start_k;

  do {
    // For now we're just linearly trying to find a thread
    thread = team->t.t_threads[k];
    k = (k + 1) % nthreads;

    // we did a full pass through all the threads
    if (k == start_k)
      pass = pass << 1;

  } while (!__kmp_give_task(thread, k, ptask, pass));

  __kmp_second_top_half_finish_proxy(taskdata);

  KA_TRACE(
      10,
      ("__kmp_proxy_task_completed_ooo(exit): proxy task completing ooo %p\n",
       taskdata));
}

// __kmp_task_dup_alloc: Allocate the taskdata and make a copy of source task
// for taskloop
//
// thread:   allocating thread
// task_src: pointer to source task to be duplicated
// returns:  a pointer to the allocated kmp_task_t structure (task).
kmp_task_t *__kmp_task_dup_alloc(kmp_info_t *thread, kmp_task_t *task_src) {
  kmp_task_t *task;
  kmp_taskdata_t *taskdata;
  kmp_taskdata_t *taskdata_src;
  kmp_taskdata_t *parent_task = thread->th.th_current_task;
  size_t shareds_offset;
  size_t task_size;

  KA_TRACE(10, ("__kmp_task_dup_alloc(enter): Th %p, source task %p\n", thread,
                task_src));
  taskdata_src = KMP_TASK_TO_TASKDATA(task_src);
  KMP_DEBUG_ASSERT(taskdata_src->td_flags.proxy ==
                   TASK_FULL); // it should not be proxy task
  KMP_DEBUG_ASSERT(taskdata_src->td_flags.tasktype == TASK_EXPLICIT);
  task_size = taskdata_src->td_size_alloc;

  // Allocate a kmp_taskdata_t block and a kmp_task_t block.
  KA_TRACE(30, ("__kmp_task_dup_alloc: Th %p, malloc size %ld\n", thread,
                task_size));
#if USE_FAST_MEMORY
  taskdata = (kmp_taskdata_t *)__kmp_fast_allocate(thread, task_size);
#else
  taskdata = (kmp_taskdata_t *)__kmp_thread_malloc(thread, task_size);
#endif /* USE_FAST_MEMORY */
  KMP_MEMCPY(taskdata, taskdata_src, task_size);

  task = KMP_TASKDATA_TO_TASK(taskdata);

  // Initialize new task (only specific fields not affected by memcpy)
  taskdata->td_task_id = KMP_GEN_TASK_ID();
  if (task->shareds != NULL) { // need setup shareds pointer
    shareds_offset = (char *)task_src->shareds - (char *)taskdata_src;
    task->shareds = &((char *)taskdata)[shareds_offset];
    KMP_DEBUG_ASSERT((((kmp_uintptr_t)task->shareds) & (sizeof(void *) - 1)) ==
                     0);
  }
  taskdata->td_alloc_thread = thread;
  taskdata->td_parent = parent_task;
  taskdata->td_taskgroup =
      parent_task
          ->td_taskgroup; // task inherits the taskgroup from the parent task

  // Only need to keep track of child task counts if team parallel and tasking
  // not serialized
  if (!(taskdata->td_flags.team_serial || taskdata->td_flags.tasking_ser)) {
    KMP_TEST_THEN_INC32(&parent_task->td_incomplete_child_tasks);
    if (parent_task->td_taskgroup){
      int tmp_count = KMP_TEST_THEN_INC32(&parent_task->td_taskgroup->count) + 1;
      //fprintf(stderr, "%f __kmp_task_dup_alloc: T#%d incr taskgroup->count to %d\n",get_wall_time2(), __kmp_entry_gtid(), tmp_count);
    }
    // Only need to keep track of allocated child tasks for explicit tasks since
    // implicit not deallocated
    if (taskdata->td_parent->td_flags.tasktype == TASK_EXPLICIT)
      KMP_TEST_THEN_INC32(&taskdata->td_parent->td_allocated_child_tasks);
  }

  KA_TRACE(20,
           ("__kmp_task_dup_alloc(exit): Th %p, created task %p, parent=%p\n",
            thread, taskdata, taskdata->td_parent));
#if OMPT_SUPPORT
  if (__builtin_expect(ompt_enabled.enabled,0)) __ompt_task_init(taskdata, thread->th.th_info.ds.ds_gtid);
#endif
  return task;
}

// Routine optionally generated by the compiler for setting the lastprivate flag
// and calling needed constructors for private/firstprivate objects
// (used to form taskloop tasks from pattern task)
// Parameters: dest task, src task, lastprivate flag.
typedef void (*p_task_dup_t)(kmp_task_t *, kmp_task_t *, kmp_int32);

// __kmp_taskloop_linear: Start tasks of the taskloop linearly
//
// loc       Source location information
// gtid      Global thread ID
// task      Pattern task, exposes the loop iteration range
// lb        Pointer to loop lower bound in task structure
// ub        Pointer to loop upper bound in task structure
// st        Loop stride
// ub_glob   Global upper bound (used for lastprivate check)
// num_tasks Number of tasks to execute
// grainsize Number of loop iterations per task
// extras    Number of chunks with grainsize+1 iterations
// tc        Iterations count
// task_dup  Tasks duplication routine
void __kmp_taskloop_linear(ident_t *loc, int gtid, kmp_task_t *task,
                           kmp_uint64 *lb, kmp_uint64 *ub, kmp_int64 st,
                           kmp_uint64 ub_glob, kmp_uint64 num_tasks,
                           kmp_uint64 grainsize, kmp_uint64 extras,
                           kmp_uint64 tc, void *task_dup) {
  KMP_COUNT_BLOCK(OMP_TASKLOOP);
  KMP_TIME_PARTITIONED_BLOCK(OMP_taskloop_scheduling);
  p_task_dup_t ptask_dup = (p_task_dup_t)task_dup;
  kmp_uint64 lower = *lb; // compiler provides global bounds here
  kmp_uint64 upper = *ub;
  kmp_uint64 i;
  kmp_info_t *thread = __kmp_threads[gtid];
  kmp_taskdata_t *current_task = thread->th.th_current_task;
  kmp_task_t *next_task;
  kmp_int32 lastpriv = 0;
  size_t lower_offset =
      (char *)lb - (char *)task; // remember offset of lb in the task structure
  size_t upper_offset =
      (char *)ub - (char *)task; // remember offset of ub in the task structure

  KMP_DEBUG_ASSERT(tc == num_tasks * grainsize + extras);
  KMP_DEBUG_ASSERT(num_tasks > extras);
  KMP_DEBUG_ASSERT(num_tasks > 0);
  KA_TRACE(20, ("__kmp_taskloop_linear: T#%d: %lld tasks, grainsize %lld, "
                "extras %lld, i=%lld,%lld(%d)%lld, dup %p\n",
                gtid, num_tasks, grainsize, extras, lower, upper, ub_glob, st,
                task_dup));

  // Launch num_tasks tasks, assign grainsize iterations each task
  for (i = 0; i < num_tasks; ++i) {
    kmp_uint64 chunk_minus_1;
    if (extras == 0) {
      chunk_minus_1 = grainsize - 1;
    } else {
      chunk_minus_1 = grainsize;
      --extras; // first extras iterations get bigger chunk (grainsize+1)
    }
    upper = lower + st * chunk_minus_1;
    if (i == num_tasks - 1) {
      // schedule the last task, set lastprivate flag if needed
      if (st == 1) { // most common case
        KMP_DEBUG_ASSERT(upper == *ub);
        if (upper == ub_glob)
          lastpriv = 1;
      } else if (st > 0) { // positive loop stride
        KMP_DEBUG_ASSERT((kmp_uint64)st > *ub - upper);
        if ((kmp_uint64)st > ub_glob - upper)
          lastpriv = 1;
      } else { // negative loop stride
        KMP_DEBUG_ASSERT(upper + st < *ub);
        if (upper - ub_glob < (kmp_uint64)(-st))
          lastpriv = 1;
      }
    }
    next_task = __kmp_task_dup_alloc(thread, task); // allocate new task
    // adjust task-specific bounds
    *(kmp_uint64 *)((char *)next_task + lower_offset) = lower;
    *(kmp_uint64 *)((char *)next_task + upper_offset) = upper;
    if (ptask_dup != NULL) // set lastprivate flag, construct fistprivates, etc.
      ptask_dup(next_task, task, lastpriv);
    KA_TRACE(40, ("__kmp_taskloop_linear: T#%d; task %p: lower %lld, "
                  "upper %lld (offsets %p %p)\n",
                  gtid, next_task, lower, upper, lower_offset, upper_offset));
    __kmp_omp_task(gtid, next_task, true); // schedule new task
    lower = upper + st; // adjust lower bound for the next iteration
  }
  // free the pattern task and exit
  __kmp_task_start(gtid, task, current_task); // make internal bookkeeping
  // do not execute the pattern task, just do internal bookkeeping
  __kmp_task_finish(gtid, task, current_task);
}

// Structure to keep taskloop parameters for auxiliary task
// kept in the shareds of the task structure.
typedef struct __taskloop_params {
  kmp_task_t *task;
  kmp_uint64 *lb;
  kmp_uint64 *ub;
  void *task_dup;
  kmp_int64 st;
  kmp_uint64 ub_glob;
  kmp_uint64 num_tasks;
  kmp_uint64 grainsize;
  kmp_uint64 extras;
  kmp_uint64 tc;
  kmp_uint64 num_t_min;
} __taskloop_params_t;

void __kmp_taskloop_recur(ident_t *, int, kmp_task_t *, kmp_uint64 *,
                          kmp_uint64 *, kmp_int64, kmp_uint64, kmp_uint64,
                          kmp_uint64, kmp_uint64, kmp_uint64, kmp_uint64,
                          void *);

// Execute part of the the taskloop submitted as a task.
int __kmp_taskloop_task(int gtid, void *ptask) {
  __taskloop_params_t *p =
      (__taskloop_params_t *)((kmp_task_t *)ptask)->shareds;
  kmp_task_t *task = p->task;
  kmp_uint64 *lb = p->lb;
  kmp_uint64 *ub = p->ub;
  void *task_dup = p->task_dup;
  //  p_task_dup_t ptask_dup = (p_task_dup_t)task_dup;
  kmp_int64 st = p->st;
  kmp_uint64 ub_glob = p->ub_glob;
  kmp_uint64 num_tasks = p->num_tasks;
  kmp_uint64 grainsize = p->grainsize;
  kmp_uint64 extras = p->extras;
  kmp_uint64 tc = p->tc;
  kmp_uint64 num_t_min = p->num_t_min;
#if KMP_DEBUG
  kmp_taskdata_t *taskdata = KMP_TASK_TO_TASKDATA(task);
  KMP_DEBUG_ASSERT(task != NULL);
  KA_TRACE(20, ("__kmp_taskloop_task: T#%d, task %p: %lld tasks, grainsize"
                " %lld, extras %lld, i=%lld,%lld(%d), dup %p\n",
                gtid, taskdata, num_tasks, grainsize, extras, *lb, *ub, st,
                task_dup));
#endif
  KMP_DEBUG_ASSERT(num_tasks * 2 + 1 > num_t_min);
  if (num_tasks > num_t_min)
    __kmp_taskloop_recur(NULL, gtid, task, lb, ub, st, ub_glob, num_tasks,
                         grainsize, extras, tc, num_t_min, task_dup);
  else
    __kmp_taskloop_linear(NULL, gtid, task, lb, ub, st, ub_glob, num_tasks,
                          grainsize, extras, tc, task_dup);

  KA_TRACE(40, ("__kmp_taskloop_task(exit): T#%d\n", gtid));
  return 0;
}

// Schedule part of the the taskloop as a task,
// execute the rest of the the taskloop.
//
// loc       Source location information
// gtid      Global thread ID
// task      Pattern task, exposes the loop iteration range
// lb        Pointer to loop lower bound in task structure
// ub        Pointer to loop upper bound in task structure
// st        Loop stride
// ub_glob   Global upper bound (used for lastprivate check)
// num_tasks Number of tasks to execute
// grainsize Number of loop iterations per task
// extras    Number of chunks with grainsize+1 iterations
// tc        Iterations count
// num_t_min Threashold to launch tasks recursively
// task_dup  Tasks duplication routine
void __kmp_taskloop_recur(ident_t *loc, int gtid, kmp_task_t *task,
                          kmp_uint64 *lb, kmp_uint64 *ub, kmp_int64 st,
                          kmp_uint64 ub_glob, kmp_uint64 num_tasks,
                          kmp_uint64 grainsize, kmp_uint64 extras,
                          kmp_uint64 tc, kmp_uint64 num_t_min, void *task_dup) {
#if KMP_DEBUG
  kmp_taskdata_t *taskdata = KMP_TASK_TO_TASKDATA(task);
  KMP_DEBUG_ASSERT(task != NULL);
  KMP_DEBUG_ASSERT(num_tasks > num_t_min);
  KA_TRACE(20, ("__kmp_taskloop_recur: T#%d, task %p: %lld tasks, grainsize"
                " %lld, extras %lld, i=%lld,%lld(%d), dup %p\n",
                gtid, taskdata, num_tasks, grainsize, extras, *lb, *ub, st,
                task_dup));
#endif
  p_task_dup_t ptask_dup = (p_task_dup_t)task_dup;
  kmp_uint64 lower = *lb;
  kmp_uint64 upper = *ub;
  kmp_info_t *thread = __kmp_threads[gtid];
  //  kmp_taskdata_t *current_task = thread->th.th_current_task;
  kmp_task_t *next_task;
  kmp_int32 lastpriv = 0;
  size_t lower_offset =
      (char *)lb - (char *)task; // remember offset of lb in the task structure
  size_t upper_offset =
      (char *)ub - (char *)task; // remember offset of ub in the task structure

  KMP_DEBUG_ASSERT(tc == num_tasks * grainsize + extras);
  KMP_DEBUG_ASSERT(num_tasks > extras);
  KMP_DEBUG_ASSERT(num_tasks > 0);

  // split the loop in two halves
  kmp_uint64 lb1, ub0, tc0, tc1, ext0, ext1;
  kmp_uint64 gr_size0 = grainsize;
  kmp_uint64 n_tsk0 = num_tasks >> 1; // num_tasks/2 to execute
  kmp_uint64 n_tsk1 = num_tasks - n_tsk0; // to schedule as a task
  if (n_tsk0 <= extras) {
    gr_size0++; // integrate extras into grainsize
    ext0 = 0; // no extra iters in 1st half
    ext1 = extras - n_tsk0; // remaining extras
    tc0 = gr_size0 * n_tsk0;
    tc1 = tc - tc0;
  } else { // n_tsk0 > extras
    ext1 = 0; // no extra iters in 2nd half
    ext0 = extras;
    tc1 = grainsize * n_tsk1;
    tc0 = tc - tc1;
  }
  ub0 = lower + st * (tc0 - 1);
  lb1 = ub0 + st;

  // create pattern task for 2nd half of the loop
  next_task = __kmp_task_dup_alloc(thread, task); // duplicate the task
  // adjust lower bound (upper bound is not changed) for the 2nd half
  *(kmp_uint64 *)((char *)next_task + lower_offset) = lb1;
  if (ptask_dup != NULL) // construct fistprivates, etc.
    ptask_dup(next_task, task, 0);
  *ub = ub0; // adjust upper bound for the 1st half

  // create auxiliary task for 2nd half of the loop
  kmp_task_t *new_task =
      __kmpc_omp_task_alloc(loc, gtid, 1, 3 * sizeof(void *),
                            sizeof(__taskloop_params_t), &__kmp_taskloop_task);
  __taskloop_params_t *p = (__taskloop_params_t *)new_task->shareds;
  p->task = next_task;
  p->lb = (kmp_uint64 *)((char *)next_task + lower_offset);
  p->ub = (kmp_uint64 *)((char *)next_task + upper_offset);
  p->task_dup = task_dup;
  p->st = st;
  p->ub_glob = ub_glob;
  p->num_tasks = n_tsk1;
  p->grainsize = grainsize;
  p->extras = ext1;
  p->tc = tc1;
  p->num_t_min = num_t_min;
  __kmp_omp_task(gtid, new_task, true); // schedule new task

  // execute the 1st half of current subrange
  if (n_tsk0 > num_t_min)
    __kmp_taskloop_recur(loc, gtid, task, lb, ub, st, ub_glob, n_tsk0, gr_size0,
                         ext0, tc0, num_t_min, task_dup);
  else
    __kmp_taskloop_linear(loc, gtid, task, lb, ub, st, ub_glob, n_tsk0,
                          gr_size0, ext0, tc0, task_dup);

  KA_TRACE(40, ("__kmpc_taskloop_recur(exit): T#%d\n", gtid));
}

/*!
@ingroup TASKING
@param loc       Source location information
@param gtid      Global thread ID
@param task      Task structure
@param if_val    Value of the if clause
@param lb        Pointer to loop lower bound in task structure
@param ub        Pointer to loop upper bound in task structure
@param st        Loop stride
@param nogroup   Flag, 1 if nogroup clause specified, 0 otherwise
@param sched     Schedule specified 0/1/2 for none/grainsize/num_tasks
@param grainsize Schedule value if specified
@param task_dup  Tasks duplication routine

Execute the taskloop construct.
*/
void __kmpc_taskloop(ident_t *loc, int gtid, kmp_task_t *task, int if_val,
                     kmp_uint64 *lb, kmp_uint64 *ub, kmp_int64 st, int nogroup,
                     int sched, kmp_uint64 grainsize, void *task_dup) {
  kmp_taskdata_t *taskdata = KMP_TASK_TO_TASKDATA(task);
  KMP_DEBUG_ASSERT(task != NULL);

  KA_TRACE(20, ("__kmpc_taskloop: T#%d, task %p, lb %lld, ub %lld, st %lld, "
                "grain %llu(%d), dup %p\n",
                gtid, taskdata, *lb, *ub, st, grainsize, sched, task_dup));

#if OMPT_SUPPORT && OMPT_OPTIONAL
  ompt_team_info_t *team_info = __ompt_get_teaminfo(0, NULL);
  ompt_task_info_t *task_info = __ompt_get_task_info_object(0);
  if (ompt_enabled.ompt_callback_work) {
    ompt_callbacks.ompt_callback(ompt_callback_work)(
        ompt_work_taskloop, ompt_scope_begin, &(team_info->parallel_data),
        &(task_info->task_data),
        0, // TODO: OMPT: verify loop count value (OpenMP-spec 4.6.2.18). ?? Should ve 'tc' value below?
        OMPT_GET_RETURN_ADDRESS(0));
  }
#endif

  if (nogroup == 0){
#if OMPT_SUPPORT && OMPT_OPTIONAL
    OMPT_STORE_RETURN_ADDRESS(gtid);
#endif
     __kmpc_taskgroup(loc, gtid);
  }

  // =========================================================================
  // calculate loop parameters
  kmp_uint64 tc;
  kmp_uint64 lower = *lb; // compiler provides global bounds here
  kmp_uint64 upper = *ub;
  kmp_uint64 ub_glob = upper; // global upper used to calc lastprivate flag
  kmp_uint64 num_tasks = 0, extras = 0;
  kmp_uint64 num_tasks_min = __kmp_taskloop_min_tasks;
  kmp_info_t *thread = __kmp_threads[gtid];
  kmp_taskdata_t *current_task = thread->th.th_current_task;

  // compute trip count
  if (st == 1) { // most common case
    tc = upper - lower + 1;
  } else if (st < 0) {
    tc = (lower - upper) / (-st) + 1;
  } else { // st > 0
    tc = (upper - lower) / st + 1;
  }
  if (tc == 0) {
    KA_TRACE(20, ("__kmpc_taskloop(exit): T#%d zero-trip loop\n", gtid));
    // free the pattern task and exit
    __kmp_task_start(gtid, task, current_task);
    // do not execute anything for zero-trip loop
    __kmp_task_finish(gtid, task, current_task);
    return;
  }
  if (num_tasks_min == 0)
    // TODO: can we choose better default heuristic?
    num_tasks_min =
        KMP_MIN(thread->th.th_team_nproc * 10, INITIAL_TASK_DEQUE_SIZE);

  // compute num_tasks/grainsize based on the input provided
  switch (sched) {
  case 0: // no schedule clause specified, we can choose the default
    // let's try to schedule (team_size*10) tasks
    grainsize = thread->th.th_team_nproc * 10;
  case 2: // num_tasks provided
    if (grainsize > tc) {
      num_tasks = tc; // too big num_tasks requested, adjust values
      grainsize = 1;
      extras = 0;
    } else {
      num_tasks = grainsize;
      grainsize = tc / num_tasks;
      extras = tc % num_tasks;
    }
    break;
  case 1: // grainsize provided
    if (grainsize > tc) {
      num_tasks = 1; // too big grainsize requested, adjust values
      grainsize = tc;
      extras = 0;
    } else {
      num_tasks = tc / grainsize;
      // adjust grainsize for balanced distribution of iterations
      grainsize = tc / num_tasks;
      extras = tc % num_tasks;
    }
    break;
  default:
    KMP_ASSERT2(0, "unknown scheduling of taskloop");
  }
  KMP_DEBUG_ASSERT(tc == num_tasks * grainsize + extras);
  KMP_DEBUG_ASSERT(num_tasks > extras);
  KMP_DEBUG_ASSERT(num_tasks > 0);
  // =========================================================================

  // check if clause value first
  if (if_val == 0) { // if(0) specified, mark task as serial
    taskdata->td_flags.task_serial = 1;
    taskdata->td_flags.tiedness = TASK_TIED; // AC: serial task cannot be untied
#if OMPT_SUPPORT && OMPT_OPTIONAL
    OMPT_STORE_RETURN_ADDRESS(gtid);
#endif
    // always start serial tasks linearly
    __kmp_taskloop_linear(loc, gtid, task, lb, ub, st, ub_glob, num_tasks,
                          grainsize, extras, tc, task_dup);
  } else if (num_tasks > num_tasks_min) {
    KA_TRACE(20, ("__kmpc_taskloop: T#%d, go recursive: tc %llu, #tasks %llu"
                  "(%lld), grain %llu, extras %llu\n",
                  gtid, tc, num_tasks, num_tasks_min, grainsize, extras));
#if OMPT_SUPPORT && OMPT_OPTIONAL
    OMPT_STORE_RETURN_ADDRESS(gtid);
#endif
    __kmp_taskloop_recur(loc, gtid, task, lb, ub, st, ub_glob, num_tasks,
                         grainsize, extras, tc, num_tasks_min, task_dup);
  } else {
    KA_TRACE(20, ("__kmpc_taskloop: T#%d, go linear: tc %llu, #tasks %llu"
                  "(%lld), grain %llu, extras %llu\n",
                  gtid, tc, num_tasks, num_tasks_min, grainsize, extras));
#if OMPT_SUPPORT && OMPT_OPTIONAL
    OMPT_STORE_RETURN_ADDRESS(gtid);
#endif
    __kmp_taskloop_linear(loc, gtid, task, lb, ub, st, ub_glob, num_tasks,
                          grainsize, extras, tc, task_dup);
  }

  if (nogroup == 0) {
#if OMPT_SUPPORT && OMPT_OPTIONAL
    OMPT_STORE_RETURN_ADDRESS(gtid);
#endif
    __kmpc_end_taskgroup(loc, gtid);
  }
#if OMPT_SUPPORT && OMPT_OPTIONAL
  if (ompt_enabled.ompt_callback_work) {
    ompt_callbacks.ompt_callback(ompt_callback_work)(
        ompt_work_taskloop, ompt_scope_end, &(team_info->parallel_data),
        &(task_info->task_data),
        0, // TODO: OMPT: verify loop count value (OpenMP-spec 4.6.2.18). ?? Should ve 'tc' value below?
        OMPT_GET_RETURN_ADDRESS(0));
  }
#endif
  KA_TRACE(20, ("__kmpc_taskloop(exit): T#%d\n", gtid));
}

#endif
