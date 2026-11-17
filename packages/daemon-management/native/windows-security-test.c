#include "windows-security.h"
#include <aclapi.h>
#include <sddl.h>
#include <stdio.h>
#include <stdlib.h>
#include <wchar.h>

static void require(BOOL condition, const char *message) {
  if (!condition) { fprintf(stderr, "%s (Windows error %lu)\n", message, GetLastError()); ExitProcess(1); }
}
static void acl(WCHAR *path, PSECURITY_DESCRIPTOR descriptor, BOOL protected) {
  PACL dacl; BOOL present, defaulted;
  require(GetSecurityDescriptorDacl(descriptor, &present, &dacl, &defaulted) && present, "extract fixture ACL");
  DWORD error = SetNamedSecurityInfoW(path, SE_FILE_OBJECT,
    DACL_SECURITY_INFORMATION | (protected ? PROTECTED_DACL_SECURITY_INFORMATION : UNPROTECTED_DACL_SECURITY_INFORMATION),
    NULL, NULL, dacl, NULL);
  SetLastError(error); require(error == ERROR_SUCCESS, "install fixture ACL");
}
static void rejects(const WCHAR *path, const char *message) {
  HANDLE file, directory;
  DWORD error = magnitude_open_private_lock(path, &file, &directory);
  require(error != ERROR_SUCCESS && file == INVALID_HANDLE_VALUE && directory == INVALID_HANDLE_VALUE, message);
}
static void extends(const WCHAR *full, const WCHAR *expected, const char *message) {
  WCHAR *extended = NULL;
  require(magnitude_extended_from_full(full, &extended) == ERROR_SUCCESS && !wcscmp(extended, expected), message);
  free(extended);
}
/* A profile deeper than MAX_PATH (the tester's 302 and 339 characters) owns its lock and private files without
 * LongPathsEnabled, which is off by default. */
static void long_profile(const WCHAR *cwd) {
  WCHAR deep[4096], lock[4096], updates[4096], key[4096];
  require(swprintf(deep, 4096, L"%ls\\long-profile", cwd) > 0, "long profile root");
  while (wcslen(deep) < 340) require(wcscat_s(deep, 4096, L"\\a-directory-name-to-pass-max-path") == 0, "deepen profile path");
  require(swprintf(lock, 4096, L"%ls\\application.lock", deep) > 0, "long lock path");
  require(swprintf(updates, 4096, L"%ls\\updates", deep) > 0, "long updates path");
  require(swprintf(key, 4096, L"%ls\\key.pem", updates) > 0, "long key path");
  WCHAR *extended = NULL; WCHAR full[4096];
  require(swprintf(full, 4096, L"\\\\?\\%ls", cwd) > 0 && magnitude_extended_from_full(full, &extended) == ERROR_SUCCESS, "extended cwd");
  WCHAR walk[4096]; wcscpy_s(walk, 4096, extended); free(extended);
  for (const WCHAR *rest = deep + wcslen(cwd); *rest; ) {
    const WCHAR *next = wcschr(rest + 1, L'\\');
    size_t segment = next ? (size_t)(next - rest) : wcslen(rest);
    wcsncat_s(walk, 4096, rest, segment);
    if (next) require(CreateDirectoryW(walk, NULL) || GetLastError() == ERROR_ALREADY_EXISTS, "create long profile parents");
    rest += segment;
  }
  HANDLE file, directory; WCHAR endpoint[128]; BOOL missing;
  require(magnitude_open_private_lock(lock, &file, &directory) == ERROR_SUCCESS, "own a lock past MAX_PATH");
  require(magnitude_inspect_application_endpoint(deep, endpoint, &missing) == ERROR_SUCCESS && !missing, "find the endpoint past MAX_PATH");
  CloseHandle(file); CloseHandle(directory);
  require(magnitude_prepare_private_directory(updates) == ERROR_SUCCESS, "private updates directory past MAX_PATH");
  require(magnitude_create_private_content(key) == ERROR_SUCCESS && magnitude_validate_private_content(key) == ERROR_SUCCESS, "private file past MAX_PATH");
}
int wmain(void) {
  WCHAR cwd[32768], directory_path[32768], lock_path[32768], moved[32768];
  DWORD length = GetCurrentDirectoryW(32768, cwd);
  require(length > 0 && length < 32000, "bounded fixture directory");
  extends(L"C:\\Users\\me\\AppData\\Local\\Magnitude", L"\\\\?\\C:\\Users\\me\\AppData\\Local\\Magnitude", "drive path gets the extended prefix");
  extends(L"\\\\server\\share\\Magnitude", L"\\\\?\\UNC\\server\\share\\Magnitude", "share path gets the extended UNC prefix");
  extends(L"\\\\?\\C:\\Magnitude", L"\\\\?\\C:\\Magnitude", "an extended path is kept");
  extends(L"\\\\.\\pipe\\magnitude", L"\\\\.\\pipe\\magnitude", "a device path is kept");
  long_profile(cwd);
  require(swprintf(directory_path, 32768, L"%ls\\private-state", cwd) > 0, "directory path");
  require(swprintf(lock_path, 32768, L"%ls\\application.lock", directory_path) > 0, "lock path");
  require(swprintf(moved, 32768, L"%ls\\renamed-state", cwd) > 0, "rename path");
  WCHAR endpoint[128], inspected[128], alias[32768]; BOOL missing;
  require(magnitude_inspect_application_endpoint(directory_path, inspected, &missing) == ERROR_SUCCESS && missing, "absent directory lookup");
  require(GetFileAttributesW(directory_path) == INVALID_FILE_ATTRIBUTES, "lookup does not create state");
  HANDLE file, directory;
  require(magnitude_open_private_lock(lock_path, &file, &directory) == ERROR_SUCCESS, "create protected directory and file");
  require(magnitude_directory_endpoint(directory, endpoint) == ERROR_SUCCESS, "retained directory endpoint");
  require(magnitude_inspect_application_endpoint(directory_path, inspected, &missing) == ERROR_SUCCESS && !missing && !wcscmp(endpoint, inspected), "owner and observer agree");
  require(swprintf(alias, 32768, L"\\\\?\\%ls", directory_path) > 0, "extended directory alias");
  require(magnitude_inspect_application_endpoint(alias, inspected, &missing) == ERROR_SUCCESS && !missing && !wcscmp(endpoint, inspected), "extended path has the same endpoint");
  CloseHandle(file);
  require(!MoveFileW(directory_path, moved), "retained directory prevents replacement");
  require(GetLastError() == ERROR_SHARING_VIOLATION, "retained directory rejects rename through sharing enforcement");
  CloseHandle(directory);
  require(MoveFileW(directory_path, moved), "released directory can be renamed");
  require(MoveFileW(moved, directory_path), "restore released fixture directory");

  PSECURITY_DESCRIPTOR private_file = NULL, private_directory = NULL, everyone = NULL;
  require(magnitude_private_descriptor(FALSE, &private_file) == ERROR_SUCCESS, "private file descriptor");
  require(magnitude_private_descriptor(TRUE, &private_directory) == ERROR_SUCCESS, "private directory descriptor");
  require(ConvertStringSecurityDescriptorToSecurityDescriptorW(L"D:P(A;;FA;;;WD)", SDDL_REVISION_1, &everyone, NULL), "broad fixture descriptor");

  WCHAR update_directory[32768], update_file[32768], update_link[32768];
  require(swprintf(update_directory, 32768, L"%ls\\updates", cwd) > 0, "update directory path");
  require(swprintf(update_file, 32768, L"%ls\\key.pem", update_directory) > 0, "update key path");
  require(swprintf(update_link, 32768, L"%ls\\key-link.pem", update_directory) > 0, "update link path");
  require(magnitude_prepare_private_directory(update_directory) == ERROR_SUCCESS, "create private updates directory");
  require(magnitude_prepare_private_directory(update_directory) == ERROR_SUCCESS, "reuse private updates directory");
  require(magnitude_create_private_content(update_file) == ERROR_SUCCESS, "create private key before writing content");
  require(magnitude_create_private_content(update_file) != ERROR_SUCCESS, "refuse to overwrite existing key");
  require(magnitude_validate_private_content(update_file) == ERROR_SUCCESS, "accept explicit current-user-only key");
  require(CreateHardLinkW(update_link, update_file, NULL), "create hard link fixture");
  require(magnitude_validate_private_content(update_file) != ERROR_SUCCESS, "reject linked key");
  require(DeleteFileW(update_link), "remove hard link fixture");
  acl(update_file, everyone, TRUE);
  require(magnitude_validate_private_content(update_file) != ERROR_SUCCESS, "reject broad key permissions");
  acl(update_file, private_file, TRUE);
  require(magnitude_validate_private_content(update_file) == ERROR_SUCCESS, "accept explicit private key");
  require(magnitude_validate_private_content(update_directory) != ERROR_SUCCESS, "reject directory as key");
  acl(update_directory, everyone, TRUE);
  require(magnitude_prepare_private_directory(update_directory) != ERROR_SUCCESS, "refuse to repair broad updates directory");
  acl(update_directory, private_directory, TRUE);
  require(DeleteFileW(update_file) && RemoveDirectoryW(update_directory), "clean update permissions fixture");

  BOOL retired = TRUE;
  require(magnitude_recover_update_directory(update_directory, &retired) == ERROR_SUCCESS && !retired, "fresh update directory needs no recovery");
  require(magnitude_recover_update_directory(update_directory, &retired) == ERROR_SUCCESS && !retired, "private update directory needs no recovery");
  acl(update_directory, everyone, TRUE);
  require(magnitude_recover_update_directory(update_directory, &retired) != ERROR_SUCCESS && !retired, "recovery refuses broad ACLs");
  acl(update_directory, private_directory, TRUE);
  require(RemoveDirectoryW(update_directory), "remove private fixture");
  /* Model the user-owned cache independently of the runner's temporary ACL and
   * elevated-token default owner. The child still inherits its actual ACL. */
  WCHAR cache_parent[32768];
  require(swprintf(cache_parent, 32768, L"%ls\\cache-parent", cwd) > 0, "cache parent path");
  require(magnitude_prepare_private_directory(cache_parent) == ERROR_SUCCESS, "create cache fixture parent");
  require(swprintf(update_directory, 32768, L"%ls\\updates", cache_parent) > 0, "inherited cache path");
  require(swprintf(update_file, 32768, L"%ls\\key.pem", update_directory) > 0, "unknown cache path");
  PSID cache_owner = NULL; BOOL owner_defaulted;
  SECURITY_DESCRIPTOR inherited_descriptor;
  require(GetSecurityDescriptorOwner(private_directory, &cache_owner, &owner_defaulted), "read fixture owner");
  WCHAR *cache_sid = NULL, cache_sddl[512]; PSECURITY_DESCRIPTOR cache_acl = NULL;
  require(ConvertSidToStringSidW(cache_owner, &cache_sid), "format cache owner");
  require(swprintf(cache_sddl, 512, L"D:P(A;OICI;FA;;;%ls)(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)", cache_sid) > 0,
    "format known cache permissions");
  LocalFree(cache_sid);
  require(ConvertStringSecurityDescriptorToSecurityDescriptorW(cache_sddl, SDDL_REVISION_1, &cache_acl, NULL),
    "create known cache ACL");
  acl(cache_parent, cache_acl, TRUE);
  LocalFree(cache_acl);
  require(InitializeSecurityDescriptor(&inherited_descriptor, SECURITY_DESCRIPTOR_REVISION) &&
    SetSecurityDescriptorOwner(&inherited_descriptor, cache_owner, FALSE), "set inherited cache owner");
  SECURITY_ATTRIBUTES inherited_attributes = { sizeof(inherited_attributes), &inherited_descriptor, FALSE };
  require(CreateDirectoryW(update_directory, &inherited_attributes), "create inherited old cache");
  HANDLE unknown = CreateFileW(update_file, GENERIC_WRITE, 0, NULL, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, NULL);
  require(unknown != INVALID_HANDLE_VALUE, "create unknown cache entry"); CloseHandle(unknown);
  require(magnitude_recover_update_directory(update_directory, &retired) != ERROR_SUCCESS && !retired, "unknown cache contents are preserved");
  require(GetFileAttributesW(update_file) != INVALID_FILE_ATTRIBUTES, "unknown file survives refusal");
  require(DeleteFileW(update_file), "remove unknown fixture");
  require(swprintf(update_file, 32768, L"%ls\\update.json", update_directory) > 0, "old record path");
  HANDLE record = CreateFileW(update_file, GENERIC_WRITE, 0, NULL, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, NULL);
  require(record != INVALID_HANDLE_VALUE, "create old cache record"); CloseHandle(record);
  DWORD recovery_error = magnitude_recover_update_directory(update_directory, &retired);
  SetLastError(recovery_error);
  require(recovery_error == ERROR_SUCCESS && retired, "retire inherited cache");
  require(magnitude_prepare_private_directory(update_directory) == ERROR_SUCCESS, "replacement has private ACL");
  require(GetFileAttributesW(update_file) == INVALID_FILE_ATTRIBUTES, "old cache record cannot authorize installation");
  require(magnitude_recover_update_directory(update_directory, &retired) == ERROR_SUCCESS && !retired, "recovery is repeatable");
  require(RemoveDirectoryW(update_directory), "remove fresh cache");
  require(swprintf(update_link, 32768, L"%ls\\updates-retired-*", cache_parent) > 0, "retired directory pattern");
  WIN32_FIND_DATAW found;
  HANDLE search = FindFirstFileW(update_link, &found);
  require(search != INVALID_HANDLE_VALUE, "old cache retained separately");
  require(swprintf(update_directory, 32768, L"%ls\\%ls", cache_parent, found.cFileName) > 0, "retired path");
  FindClose(search);
  require(swprintf(update_file, 32768, L"%ls\\update.json", update_directory) > 0, "retired record path");
  require(DeleteFileW(update_file) && RemoveDirectoryW(update_directory), "old cache contents preserved for explicit cleanup");
  require(RemoveDirectoryW(cache_parent), "remove cache fixture parent");

  acl(lock_path, everyone, TRUE);
  rejects(lock_path, "reject broad file permissions");
  acl(lock_path, private_file, TRUE);
  acl(lock_path, private_file, FALSE);
  rejects(lock_path, "reject inherited file permissions");
  acl(lock_path, private_file, TRUE);
  require(SetNamedSecurityInfoW(lock_path, SE_FILE_OBJECT, DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION,
    NULL, NULL, NULL, NULL) == ERROR_SUCCESS, "install null file DACL");
  rejects(lock_path, "reject null file DACL");
  acl(lock_path, private_file, TRUE);

  acl(directory_path, everyone, TRUE);
  rejects(lock_path, "reject broad directory permissions even with a private file");
  require(magnitude_inspect_application_endpoint(directory_path, inspected, &missing) != ERROR_SUCCESS && !missing, "unsafe lookup is not absence");
  acl(directory_path, private_directory, TRUE);
  acl(directory_path, private_directory, FALSE);
  rejects(lock_path, "reject inherited directory permissions");
  acl(directory_path, private_directory, TRUE);
  require(magnitude_open_private_lock(lock_path, &file, &directory) == ERROR_SUCCESS, "accept restored exact private permissions");
  require(magnitude_directory_endpoint(directory, inspected) == ERROR_SUCCESS && !wcscmp(endpoint, inspected), "endpoint survives lock reacquisition");
  CloseHandle(file); CloseHandle(directory);
  require(DeleteFileW(lock_path) && RemoveDirectoryW(directory_path), "clean fixture state");
  LocalFree(private_file); LocalFree(private_directory); LocalFree(everyone);
  puts("PASS private creation, retained directory, broad/inherited/null ACL rejection and restoration");
  return 0;
}
