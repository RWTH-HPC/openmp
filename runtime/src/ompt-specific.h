#ifndef OMPT_SPECIFIC_H
#define OMPT_SPECIFIC_H

#include "kmp.h"

/*****************************************************************************
 * types
 ****************************************************************************/

typedef kmp_info_t ompt_thread_t;



/*****************************************************************************
 * forward declarations
 ****************************************************************************/

void __ompt_team_assign_id(kmp_team_t *team, ompt_parallel_data_t ompt_pid);
void __ompt_thread_assign_wait_id(void *variable);

void __ompt_lw_taskteam_init(ompt_lw_taskteam_t *lwt, ompt_thread_t *thr,
                             int gtid, void *microtask,
                             ompt_parallel_data_t** ompt_pid);

void __ompt_lw_taskteam_link(ompt_lw_taskteam_t *lwt,  ompt_thread_t *thr);

ompt_lw_taskteam_t * __ompt_lw_taskteam_unlink(ompt_thread_t *thr);

ompt_id_t __ompt_parallel_id_new(int gtid);
ompt_id_t __ompt_task_id_new(int gtid);

ompt_team_info_t *__ompt_get_teaminfo(int depth, int *size);

ompt_task_info_t *__ompt_get_task_info_object(int depth);

int __ompt_get_parallel_info_internal(int ancestor_level, ompt_data_t **parallel_data, int *team_size);

int __ompt_get_task_info_internal(
    int ancestor_level,
    ompt_task_type_t *type,
    ompt_data_t **task_data,
    ompt_frame_t **task_frame,
    ompt_data_t **parallel_data,
    int *thread_num);

ompt_thread_data_t *__ompt_get_thread_data_internal();

static uint64_t __ompt_get_get_unique_id_internal();

/*****************************************************************************
 * macros
 ****************************************************************************/

#define OMPT_HAVE_WEAK_ATTRIBUTE KMP_HAVE_WEAK_ATTRIBUTE
#define OMPT_HAVE_PSAPI KMP_HAVE_PSAPI
#define OMPT_STR_MATCH(haystack, needle) __kmp_str_match(haystack, 0, needle)

//******************************************************************************
// inline functions
//******************************************************************************

inline ompt_thread_t *
ompt_get_thread_gtid(int gtid)
{
    return (gtid >= 0) ? __kmp_thread_from_gtid(gtid) : NULL;
}


inline ompt_thread_t *
ompt_get_thread()
{
    int gtid = __kmp_get_gtid();
    return ompt_get_thread_gtid(gtid);
}


inline void
ompt_set_thread_state(ompt_thread_t *thread, ompt_state_t state)
{
    thread->th.ompt_thread_info.state = state;
}


inline const char *
ompt_get_runtime_version()
{
    return &__kmp_version_lib_ver[KMP_VERSION_MAGIC_LEN];
}

#endif
