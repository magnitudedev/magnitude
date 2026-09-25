#include "windows-security.h"
#include <aclapi.h>
#include <sddl.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <wchar.h>

static DWORD current_user(TOKEN_USER **user) {
  HANDLE token; DWORD bytes = 0;
  *user = NULL;
  if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &token)) return GetLastError();
  GetTokenInformation(token, TokenUser, NULL, 0, &bytes);
  DWORD error = GetLastError();
  if (error != ERROR_INSUFFICIENT_BUFFER || !bytes) { CloseHandle(token); return ERROR_INVALID_DATA; }
  *user = malloc(bytes);
  if (!*user) { CloseHandle(token); return ERROR_NOT_ENOUGH_MEMORY; }
  if (!GetTokenInformation(token, TokenUser, *user, bytes, &bytes)) {
    error = GetLastError(); free(*user); *user = NULL;
  } else error = ERROR_SUCCESS;
  CloseHandle(token); return error;
}
DWORD magnitude_private_descriptor(BOOL directory, PSECURITY_DESCRIPTOR *descriptor) {
  TOKEN_USER *user = NULL; WCHAR *sid = NULL, *sddl = NULL;
  *descriptor = NULL;
  DWORD error = current_user(&user);
  if (error) return error;
  if (!ConvertSidToStringSidW(user->User.Sid, &sid)) { error = GetLastError(); goto cleanup; }
  size_t size = 2 * wcslen(sid) + 48;
  sddl = calloc(size, sizeof(WCHAR));
  if (!sddl) { error = ERROR_NOT_ENOUGH_MEMORY; goto cleanup; }
  if (swprintf(sddl, size, L"O:%lsD:P(A;%ls;FA;;;%ls)", sid, directory ? L"OICI" : L"", sid) < 0) {
    error = ERROR_INVALID_DATA; goto cleanup;
  }
  if (!ConvertStringSecurityDescriptorToSecurityDescriptorW(sddl, SDDL_REVISION_1, descriptor, NULL)) error = GetLastError();
cleanup:
  free(sddl); if (sid) LocalFree(sid); free(user); return error;
}
static DWORD private_handle(HANDLE handle, BOOL directory) {
  PSID owner = NULL; PACL acl = NULL; PSECURITY_DESCRIPTOR descriptor = NULL;
  TOKEN_USER *user = NULL;
  DWORD error = GetSecurityInfo(handle, SE_FILE_OBJECT, OWNER_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION,
    &owner, NULL, &acl, NULL, &descriptor);
  if (error) return error;
  error = current_user(&user);
  if (error) goto cleanup;
  SECURITY_DESCRIPTOR_CONTROL control; DWORD revision; ACCESS_ALLOWED_ACE *ace = NULL;
  BYTE flags = (BYTE)(directory ? OBJECT_INHERIT_ACE | CONTAINER_INHERIT_ACE : 0);
  if (!owner || !IsValidSid(owner) || !EqualSid(owner, user->User.Sid) ||
      !GetSecurityDescriptorControl(descriptor, &control, &revision) || !(control & SE_DACL_PROTECTED) ||
      !acl || !IsValidAcl(acl) || acl->AceCount != 1 || !GetAce(acl, 0, (void **)&ace) ||
      ace->Header.AceType != ACCESS_ALLOWED_ACE_TYPE || ace->Header.AceFlags != flags ||
      ace->Mask != FILE_ALL_ACCESS || !IsValidSid(&ace->SidStart) || !EqualSid(&ace->SidStart, user->User.Sid)) error = ERROR_ACCESS_DENIED;
cleanup:
  free(user); LocalFree(descriptor); return error;
}
DWORD magnitude_validate_private_directory(HANDLE directory) {
  BY_HANDLE_FILE_INFORMATION info;
  if (GetFileType(directory) != FILE_TYPE_DISK || !GetFileInformationByHandle(directory, &info) ||
      !(info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) || (info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT)) return ERROR_ACCESS_DENIED;
  return private_handle(directory, TRUE);
}
DWORD magnitude_prepare_private_directory(const WCHAR *path) {
  PSECURITY_DESCRIPTOR descriptor = NULL;
  DWORD error = magnitude_private_descriptor(TRUE, &descriptor);
  if (error) return error;
  SECURITY_ATTRIBUTES attributes = { (DWORD)sizeof(attributes), descriptor, FALSE };
  if (!CreateDirectoryW(path, &attributes) && GetLastError() != ERROR_ALREADY_EXISTS) error = GetLastError();
  LocalFree(descriptor);
  if (error) return error;
  HANDLE directory = CreateFileW(path, READ_CONTROL | FILE_READ_ATTRIBUTES | FILE_LIST_DIRECTORY,
    FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING,
    FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT, NULL);
  if (directory == INVALID_HANDLE_VALUE) return GetLastError();
  error = magnitude_validate_private_directory(directory);
  CloseHandle(directory);
  return error;
}
/* Only the known inherited cache ACL can be retired. Generic private-directory
 * validation remains strict; unknown contents and permissions are preserved. */
static DWORD inherited_update_directory(HANDLE directory) {
  PSID owner = NULL; PACL acl = NULL; PSECURITY_DESCRIPTOR descriptor = NULL;
  TOKEN_USER *user = NULL;
  DWORD error = GetSecurityInfo(directory, SE_FILE_OBJECT, OWNER_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION,
    &owner, NULL, &acl, NULL, &descriptor);
  if (error) return error;
  error = current_user(&user);
  if (error) goto cleanup;
  SECURITY_DESCRIPTOR_CONTROL control; DWORD revision; BOOL has_user = FALSE;
  if (!owner || !IsValidSid(owner) || !EqualSid(owner, user->User.Sid) || !acl || !IsValidAcl(acl) ||
      !GetSecurityDescriptorControl(descriptor, &control, &revision) || (control & SE_DACL_PROTECTED) ||
      !acl->AceCount || acl->AceCount > 3) { error = ERROR_ACCESS_DENIED; goto cleanup; }
  for (DWORD index = 0; index < acl->AceCount; ++index) {
    ACCESS_ALLOWED_ACE *ace = NULL;
    if (!GetAce(acl, index, (void **)&ace) || ace->Header.AceType != ACCESS_ALLOWED_ACE_TYPE ||
        ace->Header.AceFlags != (INHERITED_ACE | OBJECT_INHERIT_ACE | CONTAINER_INHERIT_ACE) ||
        ace->Mask != FILE_ALL_ACCESS || !IsValidSid(&ace->SidStart)) { error = ERROR_ACCESS_DENIED; goto cleanup; }
    if (EqualSid(&ace->SidStart, user->User.Sid)) has_user = TRUE;
    else if (!IsWellKnownSid(&ace->SidStart, WinLocalSystemSid) &&
             !IsWellKnownSid(&ace->SidStart, WinBuiltinAdministratorsSid)) { error = ERROR_ACCESS_DENIED; goto cleanup; }
  }
  if (!has_user) error = ERROR_ACCESS_DENIED;
cleanup:
  free(user); LocalFree(descriptor); return error;
}
static BOOL update_cache_entry(const WCHAR *name, DWORD attributes) {
  if (!wcscmp(name, L".") || !wcscmp(name, L"..")) return TRUE;
  if (attributes & FILE_ATTRIBUTE_REPARSE_POINT) return FALSE;
  if (attributes & FILE_ATTRIBUTE_DIRECTORY) {
    const WCHAR *prefix = L"desktop-update-";
    size_t length = wcslen(prefix);
    if (wcsncmp(name, prefix, length) || !name[length]) return FALSE;
    for (const WCHAR *p = name + length; *p; ++p)
      if (!((*p >= L'a' && *p <= L'z') || (*p >= L'A' && *p <= L'Z') || (*p >= L'0' && *p <= L'9'))) return FALSE;
    return TRUE;
  }
  if (!wcscmp(name, L"update.json") || !wcscmp(name, L"magnitude-setup.exe")) return TRUE;
  const WCHAR *id = !wcsncmp(name, L"update-", 7) ? name + 7 : !wcsncmp(name, L"installer-", 10) ? name + 10 : NULL;
  if (!id || wcslen(id) != 40 || wcscmp(id + 36, L".tmp")) return FALSE;
  for (size_t i = 0; i < 36; ++i) {
    BOOL hyphen = i == 8 || i == 13 || i == 18 || i == 23;
    if (hyphen ? id[i] != L'-' : !((id[i] >= L'0' && id[i] <= L'9') || (id[i] >= L'a' && id[i] <= L'f'))) return FALSE;
  }
  return TRUE;
}
DWORD magnitude_recover_update_directory(const WCHAR *path, BOOL *retired) {
  *retired = FALSE;
  DWORD error = magnitude_prepare_private_directory(path);
  if (!error) return ERROR_SUCCESS;
  HANDLE directory = CreateFileW(path, READ_CONTROL | FILE_READ_ATTRIBUTES | FILE_LIST_DIRECTORY | DELETE,
    FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING,
    FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT, NULL);
  if (directory == INVALID_HANDLE_VALUE) return GetLastError();
  BY_HANDLE_FILE_INFORMATION info;
  if (GetFileType(directory) != FILE_TYPE_DISK || !GetFileInformationByHandle(directory, &info) ||
      !(info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) || (info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT)) {
    error = ERROR_ACCESS_DENIED; goto cleanup;
  }
  error = inherited_update_directory(directory);
  if (error) goto cleanup;
  BYTE buffer[65536]; DWORD count = 0;
  for (;;) {
    if (!GetFileInformationByHandleEx(directory, FileIdBothDirectoryInfo, buffer, sizeof(buffer))) {
      error = GetLastError() == ERROR_NO_MORE_FILES ? ERROR_SUCCESS : GetLastError(); break;
    }
    FILE_ID_BOTH_DIR_INFO *entry = (FILE_ID_BOTH_DIR_INFO *)buffer;
    for (;;) {
      WCHAR name[256];
      if (++count > 1024 || entry->FileNameLength >= sizeof(name) || entry->FileNameLength % sizeof(WCHAR)) {
        error = ERROR_INVALID_DATA; goto cleanup;
      }
      memcpy(name, entry->FileName, entry->FileNameLength); name[entry->FileNameLength / sizeof(WCHAR)] = 0;
      if (!update_cache_entry(name, entry->FileAttributes)) { error = ERROR_ACCESS_DENIED; goto cleanup; }
      if (!entry->NextEntryOffset) break;
      entry = (FILE_ID_BOTH_DIR_INFO *)((BYTE *)entry + entry->NextEntryOffset);
    }
  }
  if (error) goto cleanup;
  FILE_ID_INFO identity;
  if (!GetFileInformationByHandleEx(directory, FileIdInfo, &identity, sizeof(identity))) { error = GetLastError(); goto cleanup; }
  size_t length = wcslen(path);
  WCHAR *destination = calloc(length + 42, sizeof(WCHAR));
  if (!destination) { error = ERROR_NOT_ENOUGH_MEMORY; goto cleanup; }
  memcpy(destination, path, length * sizeof(WCHAR));
  memcpy(destination + length, L"-retired-", 9 * sizeof(WCHAR));
  static const WCHAR hex[] = L"0123456789abcdef";
  for (size_t i = 0; i < 16; ++i) {
    destination[length + 9 + i * 2] = hex[identity.FileId.Identifier[i] >> 4];
    destination[length + 10 + i * 2] = hex[identity.FileId.Identifier[i] & 15];
  }
  DWORD bytes = (DWORD)(wcslen(destination) * sizeof(WCHAR));
  DWORD size = (DWORD)sizeof(FILE_RENAME_INFO) + bytes;
  FILE_RENAME_INFO *rename = calloc(1, size);
  if (!rename) { free(destination); error = ERROR_NOT_ENOUGH_MEMORY; goto cleanup; }
  rename->FileNameLength = bytes;
  memcpy(rename->FileName, destination, bytes);
  if (!SetFileInformationByHandle(directory, FileRenameInfo, rename, size)) error = GetLastError();
  else *retired = TRUE;
  free(rename); free(destination);
cleanup:
  CloseHandle(directory);
  if (!error) error = magnitude_prepare_private_directory(path);
  return error;
}
DWORD magnitude_create_private_content(const WCHAR *path) {
  PSECURITY_DESCRIPTOR descriptor = NULL;
  DWORD error = magnitude_private_descriptor(FALSE, &descriptor);
  if (error) return error;
  SECURITY_ATTRIBUTES attributes = { (DWORD)sizeof(attributes), descriptor, FALSE };
  HANDLE file = CreateFileW(path, GENERIC_READ | GENERIC_WRITE, 0, &attributes,
    CREATE_NEW, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT, NULL);
  error = file == INVALID_HANDLE_VALUE ? GetLastError() : ERROR_SUCCESS;
  if (file != INVALID_HANDLE_VALUE) CloseHandle(file);
  LocalFree(descriptor); return error;
}
DWORD magnitude_validate_private_content(const WCHAR *path) {
  HANDLE file = CreateFileW(path, GENERIC_READ | READ_CONTROL, FILE_SHARE_READ, NULL,
    OPEN_EXISTING, FILE_FLAG_OPEN_REPARSE_POINT, NULL);
  if (file == INVALID_HANDLE_VALUE) return GetLastError();
  BY_HANDLE_FILE_INFORMATION info;
  DWORD error = GetFileType(file) != FILE_TYPE_DISK || !GetFileInformationByHandle(file, &info) ||
    info.nNumberOfLinks != 1 || (info.dwFileAttributes & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT))
    ? ERROR_ACCESS_DENIED : private_handle(file, FALSE);
  CloseHandle(file); return error;
}
DWORD magnitude_directory_endpoint(HANDLE directory, WCHAR endpoint[128]) {
  endpoint[0] = 0;
  DWORD error = magnitude_validate_private_directory(directory);
  if (error) return error;
  FILE_ID_INFO identity;
  if (!GetFileInformationByHandleEx(directory, FileIdInfo, &identity, (DWORD)sizeof(identity))) return GetLastError();
  BOOL identified = FALSE;
  for (size_t index = 0; index < 16; ++index) if (identity.FileId.Identifier[index]) identified = TRUE;
  if (!identified) return ERROR_INVALID_DATA;
  WCHAR *path = calloc(32768, sizeof(WCHAR));
  if (!path) return ERROR_NOT_ENOUGH_MEMORY;
  DWORD length = GetFinalPathNameByHandleW(directory, path, 32768, VOLUME_NAME_GUID | FILE_NAME_NORMALIZED);
  if (!length) { error = GetLastError(); free(path); return error; }
  if (length >= 32768 || length < 49 || wcsncmp(path, L"\\\\?\\Volume{", 11) || path[47] != L'}' || path[48] != L'\\') {
    free(path); return ERROR_NOT_SUPPORTED;
  }
  WCHAR volume[37], file[33];
  static const WCHAR hex[] = L"0123456789abcdef";
  for (size_t index = 0; index < 36; ++index) {
    WCHAR value = path[11 + index];
    if (value >= L'A' && value <= L'F') value = (WCHAR)(value + (L'a' - L'A'));
    BOOL hyphen = index == 8 || index == 13 || index == 18 || index == 23;
    if (hyphen ? value != L'-' : !((value >= L'0' && value <= L'9') || (value >= L'a' && value <= L'f'))) {
      free(path); return ERROR_INVALID_DATA;
    }
    volume[index] = value;
  }
  volume[36] = 0; free(path);
  for (size_t index = 0; index < 16; ++index) {
    file[2 * index] = hex[identity.FileId.Identifier[index] >> 4];
    file[2 * index + 1] = hex[identity.FileId.Identifier[index] & 15];
  }
  file[32] = 0;
  return swprintf(endpoint, 128, L"\\\\.\\pipe\\magnitude-app-%ls-%ls", volume, file) > 0 ? ERROR_SUCCESS : ERROR_INVALID_DATA;
}
DWORD magnitude_inspect_application_endpoint(const WCHAR *path, WCHAR endpoint[128], BOOL *missing) {
  *missing = FALSE; endpoint[0] = 0;
  HANDLE directory = CreateFileW(path, READ_CONTROL | FILE_READ_ATTRIBUTES | FILE_LIST_DIRECTORY, FILE_SHARE_READ | FILE_SHARE_WRITE,
    NULL, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT, NULL);
  if (directory == INVALID_HANDLE_VALUE) {
    DWORD error = GetLastError();
    if (error == ERROR_FILE_NOT_FOUND || error == ERROR_PATH_NOT_FOUND) { *missing = TRUE; return ERROR_SUCCESS; }
    return error;
  }
  DWORD error = magnitude_directory_endpoint(directory, endpoint);
  CloseHandle(directory); return error;
}
DWORD magnitude_open_private_lock(const WCHAR *path, HANDLE *file, HANDLE *directory) {
  *file = INVALID_HANDLE_VALUE; *directory = INVALID_HANDLE_VALUE;
  size_t length = wcslen(path);
  WCHAR *parent = calloc(length + 1, sizeof(WCHAR));
  if (!parent) return ERROR_NOT_ENOUGH_MEMORY;
  memcpy(parent, path, (length + 1) * sizeof(WCHAR));
  for (size_t index = 0; index < length; ++index) if (parent[index] == L'/') parent[index] = L'\\';
  WCHAR *separator = wcsrchr(parent, L'\\');
  if (!separator || !separator[1] || separator == parent + 2 ||
      (separator == parent + 6 && !wcsncmp(parent, L"\\\\?\\", 4))) { free(parent); return ERROR_INVALID_NAME; }
  *separator = 0;
  PSECURITY_DESCRIPTOR descriptor = NULL;
  DWORD error = magnitude_private_descriptor(TRUE, &descriptor);
  if (error) { free(parent); return error; }
  SECURITY_ATTRIBUTES attributes = { (DWORD)sizeof(attributes), descriptor, FALSE };
  if (!CreateDirectoryW(parent, &attributes) && GetLastError() != ERROR_ALREADY_EXISTS) { error = GetLastError(); goto cleanup; }
  // Metadata-only opens do not participate in Windows sharing checks. Directory
  // read access makes the omitted FILE_SHARE_DELETE prevent renaming this identity.
  *directory = CreateFileW(parent, READ_CONTROL | FILE_READ_ATTRIBUTES | FILE_LIST_DIRECTORY, FILE_SHARE_READ | FILE_SHARE_WRITE,
    NULL, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT, NULL);
  if (*directory == INVALID_HANDLE_VALUE) { error = GetLastError(); goto cleanup; }
  BY_HANDLE_FILE_INFORMATION info;
  error = magnitude_validate_private_directory(*directory);
  if (error) goto cleanup;
  LocalFree(descriptor); descriptor = NULL;
  error = magnitude_private_descriptor(FALSE, &descriptor);
  if (error) goto cleanup;
  attributes.lpSecurityDescriptor = descriptor;
  *file = CreateFileW(path, GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
    &attributes, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT, NULL);
  if (*file == INVALID_HANDLE_VALUE) { error = GetLastError(); goto cleanup; }
  if (GetFileType(*file) != FILE_TYPE_DISK || !GetFileInformationByHandle(*file, &info) || info.nNumberOfLinks != 1 ||
      (info.dwFileAttributes & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT))) { error = ERROR_ACCESS_DENIED; goto cleanup; }
  error = private_handle(*file, FALSE);
cleanup:
  if (error) {
    if (*file != INVALID_HANDLE_VALUE) CloseHandle(*file);
    if (*directory != INVALID_HANDLE_VALUE) CloseHandle(*directory);
    *file = INVALID_HANDLE_VALUE; *directory = INVALID_HANDLE_VALUE;
  }
  if (descriptor) LocalFree(descriptor); free(parent); return error;
}
