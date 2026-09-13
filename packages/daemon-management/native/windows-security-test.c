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
