/* Read-only memory sampling. Runs off the Electron thread and never acquires termination rights. */
#include <node_api.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#ifdef _WIN32
#include <windows.h>
#include <tlhelp32.h>
#include <psapi.h>
#else
#include <unistd.h>
#ifdef __APPLE__
#include <libproc.h>
#include <sys/resource.h>
#else
#include <dirent.h>
#endif
#endif

#define MAX_PROCESSES 65536
#define MAX_APPLICATION_PROCESSES 512
/* Decimal integers through this bound are exact in JavaScript. */
#define MAX_SAFE_BYTES UINT64_C(9007199254740991)
typedef struct { uint32_t pid, parent; uint64_t started; } process_identity;
typedef struct {
  napi_async_work work;
  napi_deferred deferred;
  uint32_t root;
  uint64_t bytes;
  unsigned count;
  const char *error;
} memory_work;

#ifdef _WIN32
static int identity(uint32_t pid, process_identity *result) {
  HANDLE process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
  if (!process) return 0;
  FILETIME created, exited, kernel, user;
  int ok = GetProcessTimes(process, &created, &exited, &kernel, &user) != 0;
  if (ok) { result->pid = pid; result->started = ((uint64_t)created.dwHighDateTime << 32) | created.dwLowDateTime; }
  CloseHandle(process); return ok;
}
static int process_list(process_identity *rows, unsigned *count) {
  HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
  if (snapshot == INVALID_HANDLE_VALUE) return 0;
  PROCESSENTRY32W entry = {0}; entry.dwSize = sizeof(entry);
  int ok = Process32FirstW(snapshot, &entry) != 0;
  while (ok) {
    if (*count == MAX_PROCESSES) { CloseHandle(snapshot); return 0; }
    rows[*count].pid = entry.th32ProcessID; rows[*count].parent = entry.th32ParentProcessID;
    (*count)++;
    if (!Process32NextW(snapshot, &entry)) { ok = GetLastError() == ERROR_NO_MORE_FILES; break; }
  }
  CloseHandle(snapshot); return ok;
}
static int read_memory(const process_identity *row, uint64_t *bytes) {
  HANDLE process = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, row->pid);
  if (!process) return 0;
  FILETIME created, exited, kernel, user;
  /* PROCESS_MEMORY_COUNTERS_EX2 layout, also buildable with older SDK headers.
     Unsupported operating systems fail this capability instead of substituting commit or RSS. */
  struct { PROCESS_MEMORY_COUNTERS_EX base; SIZE_T privateWorkingSet; ULONG64 sharedCommit; } counters = {0};
  counters.base.cb = sizeof(counters);
  counters.privateWorkingSet = (SIZE_T)-1;
  int ok = GetProcessTimes(process, &created, &exited, &kernel, &user) &&
    ((((uint64_t)created.dwHighDateTime << 32) | created.dwLowDateTime) == row->started) &&
    GetProcessMemoryInfo(process, (PROCESS_MEMORY_COUNTERS *)&counters, sizeof(counters)) &&
    counters.privateWorkingSet != (SIZE_T)-1;
  if (ok) *bytes = (uint64_t)counters.privateWorkingSet;
  CloseHandle(process); return ok;
}
#define MEMORY_METRIC "PrivateWorkingSet"
#elif defined(__APPLE__)
static int identity(uint32_t pid, process_identity *result) {
  struct proc_bsdinfo info;
  if (proc_pidinfo((int)pid, PROC_PIDTBSDINFO, 0, &info, sizeof(info)) != sizeof(info)) return 0;
  result->pid = pid; result->parent = info.pbi_ppid;
  result->started = info.pbi_start_tvsec * UINT64_C(1000000) + info.pbi_start_tvusec;
  return 1;
}
static int process_list(process_identity *rows, unsigned *count) {
  int *pids = calloc(MAX_PROCESSES, sizeof(int));
  if (!pids) return 0;
  int size = proc_listallpids(pids, MAX_PROCESSES * (int)sizeof(int));
  if (size <= 0 || size >= MAX_PROCESSES) { free(pids); return 0; }
  for (int i = 0; i < size; i++) if (pids[i] > 0 && identity((uint32_t)pids[i], &rows[*count])) (*count)++;
  free(pids); return 1;
}
static int read_memory(const process_identity *row, uint64_t *bytes) {
  struct rusage_info_v2 usage;
  if (proc_pid_rusage((int)row->pid, RUSAGE_INFO_V2, (rusage_info_t *)&usage) != 0) return 0;
  *bytes = usage.ri_phys_footprint; return 1;
}
#define MEMORY_METRIC "PhysicalFootprint"
#else
static int identity(uint32_t pid, process_identity *result) {
  char path[64], buffer[4096];
  snprintf(path, sizeof(path), "/proc/%u/stat", pid);
  FILE *file = fopen(path, "r"); if (!file) return 0;
  size_t size = fread(buffer, 1, sizeof(buffer) - 1, file); fclose(file);
  if (!size || size == sizeof(buffer) - 1) return 0;
  buffer[size] = 0;
  char *tail = strrchr(buffer, ')'); if (!tail || tail[1] != ' ') return 0;
  char *save = NULL, *value = strtok_r(tail + 2, " ", &save);
  uint64_t parent = 0, started = 0;
  for (int field = 3; field <= 22; field++) {
    if (!value) return 0;
    if (field == 4 || field == 22) {
      char *end; errno = 0; unsigned long long number = strtoull(value, &end, 10);
      if (errno || *end || end == value) return 0;
      if (field == 4) parent = number; else started = number;
    }
    value = strtok_r(NULL, " ", &save);
  }
  if (parent > UINT32_MAX) return 0;
  result->pid = pid; result->parent = (uint32_t)parent; result->started = started; return 1;
}
static int process_list(process_identity *rows, unsigned *count) {
  DIR *directory = opendir("/proc"); if (!directory) return 0;
  struct dirent *entry;
  while ((entry = readdir(directory))) {
    char *end; unsigned long pid = strtoul(entry->d_name, &end, 10);
    if (*end || end == entry->d_name || !pid || pid > UINT32_MAX) continue;
    if (*count == MAX_PROCESSES) { closedir(directory); return 0; }
    if (identity((uint32_t)pid, &rows[*count])) (*count)++;
  }
  closedir(directory); return 1;
}
static int read_memory(const process_identity *row, uint64_t *bytes) {
  char path[64], line[1024];
  snprintf(path, sizeof(path), "/proc/%u/smaps_rollup", row->pid);
  FILE *file = fopen(path, "r"); if (!file) return 0;
  int found = 0; unsigned lines = 0;
  while (lines++ < 256 && fgets(line, sizeof(line), file)) {
    unsigned long long kib;
    if (sscanf(line, "Pss: %llu kB", &kib) == 1 && kib <= MAX_SAFE_BYTES / 1024) {
      *bytes = (uint64_t)kib * 1024; found = 1; break;
    }
  }
  fclose(file); return found;
}
#define MEMORY_METRIC "ProportionalResident"
#endif

static void sample(napi_env env, void *data) {
  (void)env; memory_work *work = data;
  process_identity *rows = calloc(MAX_PROCESSES, sizeof(*rows));
  process_identity selected[MAX_APPLICATION_PROCESSES]; unsigned total = 0, count = 0;
  if (!rows || !process_list(rows, &total)) { work->error = "Could not inspect Magnitude's processes."; free(rows); return; }
  for (unsigned i = 0; i < total; i++) if (rows[i].pid == work->root) { selected[count++] = rows[i]; break; }
  if (!count) { work->error = "Magnitude's process identity is unavailable."; free(rows); return; }
  for (unsigned at = 0; at < count; at++) {
    for (unsigned i = 0; i < total; i++) if (rows[i].parent == selected[at].pid && rows[i].pid != selected[at].pid) {
      int seen = 0; for (unsigned j = 0; j < count; j++) if (selected[j].pid == rows[i].pid) seen = 1;
      if (seen) continue;
      if (count == MAX_APPLICATION_PROCESSES) { work->error = "Magnitude's process count exceeds the sampling limit."; goto done; }
      selected[count++] = rows[i];
    }
  }
#ifdef _WIN32
  /* Retain creation evidence before memory reads; the second snapshot fences ancestry below. */
  for (unsigned i = 0; i < count; i++) if (!identity(selected[i].pid, &selected[i])) {
    work->error = "Magnitude's processes changed while reading memory."; goto done;
  }
#endif
  for (unsigned i = 1; i < count; i++) for (unsigned j = 0; j < count; j++) {
    if (selected[i].parent == selected[j].pid && selected[i].started < selected[j].started) {
      work->error = "Magnitude's process ancestry changed while reading memory."; goto done;
    }
  }
  for (unsigned i = 0; i < count; i++) {
    process_identity before = {0}, after = {0}; uint64_t bytes = 0;
    if (!identity(selected[i].pid, &before) || before.started != selected[i].started ||
#ifndef _WIN32
        before.parent != selected[i].parent ||
#endif
        !read_memory(&selected[i], &bytes) || !identity(selected[i].pid, &after) || after.started != before.started ||
#ifndef _WIN32
        after.parent != before.parent ||
#endif
        bytes > MAX_SAFE_BYTES - work->bytes) {
      work->error = "Memory reading unavailable. Magnitude will try again shortly."; goto done;
    }
    work->bytes += bytes;
  }
#ifdef _WIN32
  total = 0;
  if (!process_list(rows, &total)) { work->error = "Could not verify Magnitude's processes."; goto done; }
  for (unsigned i = 0; i < count; i++) {
    int matched = 0;
    for (unsigned j = 0; j < total; j++) if (selected[i].pid == rows[j].pid && selected[i].parent == rows[j].parent) matched = 1;
    if (!matched) { work->error = "Magnitude's processes changed while reading memory."; goto done; }
  }
#endif
  for (unsigned i = 0; i < count; i++) {
    process_identity current = {0};
    if (!identity(selected[i].pid, &current) || current.started != selected[i].started
#ifndef _WIN32
        || current.parent != selected[i].parent
#endif
    ) { work->error = "Magnitude's processes changed while reading memory."; goto done; }
  }
  work->count = count;
done:
  free(rows);
}
static void completed(napi_env env, napi_status status, void *data) {
  memory_work *work = data; napi_value result, value;
  if (status != napi_ok || work->error) {
    napi_value message;
    napi_create_string_utf8(env, work->error ? work->error : "Memory sampling was interrupted.", NAPI_AUTO_LENGTH, &message);
    napi_create_error(env, NULL, message, &result); napi_reject_deferred(env, work->deferred, result);
  } else {
    napi_create_object(env, &result);
    napi_create_double(env, (double)work->bytes, &value); napi_set_named_property(env, result, "bytes", value);
    napi_create_uint32(env, work->count, &value); napi_set_named_property(env, result, "processCount", value);
    napi_create_string_utf8(env, MEMORY_METRIC, NAPI_AUTO_LENGTH, &value); napi_set_named_property(env, result, "metric", value);
    napi_resolve_deferred(env, work->deferred, result);
  }
  napi_delete_async_work(env, work->work); free(work);
}
static napi_value observe(napi_env env, napi_callback_info info) {
  (void)info; memory_work *work = calloc(1, sizeof(*work)); napi_value promise, name;
  if (!work) { napi_throw_error(env, NULL, "Could not allocate memory observation."); return NULL; }
#ifdef _WIN32
  work->root = GetCurrentProcessId();
#else
  work->root = (uint32_t)getpid();
#endif
  napi_create_promise(env, &work->deferred, &promise);
  napi_create_string_utf8(env, "Magnitude application memory", NAPI_AUTO_LENGTH, &name);
  if (napi_create_async_work(env, NULL, name, sample, completed, work, &work->work) != napi_ok) {
    free(work); napi_throw_error(env, NULL, "Could not create memory observation."); return NULL;
  }
  if (napi_queue_async_work(env, work->work) != napi_ok) {
    napi_delete_async_work(env, work->work); free(work); napi_throw_error(env, NULL, "Could not queue memory observation."); return NULL;
  }
  return promise;
}
void magnitude_register_application_memory(napi_env env, napi_value exports) {
  napi_property_descriptor method = {"applicationMemory", NULL, observe, NULL, NULL, NULL, napi_default, NULL};
  napi_define_properties(env, exports, 1, &method);
}
