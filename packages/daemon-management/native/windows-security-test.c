#include "windows-security.h"
#include <aclapi.h>
#include <sddl.h>
#include <stdio.h>
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
int wmain(void) {
  WCHAR cwd[32768], directory_path[32768], lock_path[32768], moved[32768];
  DWORD length = GetCurrentDirectoryW(32768, cwd);
  require(length > 0 && length < 32000, "bounded fixture directory");
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
