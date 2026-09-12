#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#include <windows.h>
#include <taskschd.h>
#include <sddl.h>
#include <bcrypt.h>
#include <cstdio>
#include <string>
#include <vector>
#include <new>
extern "C" {
#include "windows-job.h"
}

// COM may wait on the scheduler service. This executable runs inside a
// bounded parent-owned job; no COM call or cancellation wait runs in Electron.
template<class T> struct ComRef {
  T *value = nullptr;
  ~ComRef() { if (value) value->Release(); }
  ComRef() = default;
  ComRef(const ComRef &) = delete;
  ComRef &operator=(const ComRef &) = delete;
};
struct Bstr {
  BSTR value = nullptr;
  ~Bstr() { SysFreeString(value); }
};
struct ComApartment { ~ComApartment() { CoUninitialize(); } };

static std::string json(const std::string &text) {
  static const char hex[] = "0123456789abcdef";
  std::string result = "\"";
  for (const char raw : text) {
    const auto byte = static_cast<unsigned char>(raw);
    if (byte == '"' || byte == '\\') { result += '\\'; result += raw; }
    else if (byte < 0x20) {
      result += "\\u00"; result += hex[byte >> 4]; result += hex[byte & 15];
    } else result += raw;
  }
  result += '"';
  return result;
}
static HRESULT utf8(const WCHAR *text, UINT length, std::string &result) {
  if (!text || length == 0 || length > 65536) return E_INVALIDARG;
  for (UINT index = 0; index < length; ++index) if (text[index] == 0) return E_INVALIDARG;
  const int size = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, text,
    static_cast<int>(length), nullptr, 0, nullptr, nullptr);
  if (!size) return HRESULT_FROM_WIN32(GetLastError());
  result.resize(static_cast<size_t>(size));
  if (!WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, text,
      static_cast<int>(length), result.data(), size, nullptr, nullptr))
    return HRESULT_FROM_WIN32(GetLastError());
  return S_OK;
}
static HRESULT current_sid(std::string &result) {
  HANDLE token = nullptr;
  if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &token)) return HRESULT_FROM_WIN32(GetLastError());
  struct Token { HANDLE value; ~Token() { CloseHandle(value); } } retained{token};
  DWORD size = 0;
  if (GetTokenInformation(token, TokenUser, nullptr, 0, &size) ||
      GetLastError() != ERROR_INSUFFICIENT_BUFFER || size == 0 || size > 65536) return E_UNEXPECTED;
  std::vector<unsigned char> bytes(size);
  if (!GetTokenInformation(token, TokenUser, bytes.data(), size, &size)) return HRESULT_FROM_WIN32(GetLastError());
  WCHAR *sid = nullptr;
  if (!ConvertSidToStringSidW(reinterpret_cast<TOKEN_USER *>(bytes.data())->User.Sid, &sid))
    return HRESULT_FROM_WIN32(GetLastError());
  const HRESULT status = utf8(sid, static_cast<UINT>(wcslen(sid)), result);
  LocalFree(sid);
  return status;
}
static HRESULT digest_matches(const std::string &document, const WCHAR *expected) {
  if (!expected || wcslen(expected) != 64) return E_INVALIDARG;
  unsigned char digest[32];
  const NTSTATUS status = BCryptHash(BCRYPT_SHA256_ALG_HANDLE, nullptr, 0,
    reinterpret_cast<PUCHAR>(const_cast<char *>(document.data())),
    static_cast<ULONG>(document.size()), digest, sizeof(digest));
  if (status < 0) return HRESULT_FROM_NT(status);
  static const WCHAR hex[] = L"0123456789abcdef";
  for (size_t index = 0; index < sizeof(digest); ++index) {
    if (expected[index * 2] != hex[digest[index] >> 4] ||
        expected[index * 2 + 1] != hex[digest[index] & 15])
      return HRESULT_FROM_WIN32(ERROR_REVISION_MISMATCH);
  }
  return S_OK;
}
static HRESULT query(std::string &output, const WCHAR *expected_digest, const WCHAR *expected_sid) {
  const DWORD containment = magnitude_owned_validate_current();
  if (containment) return HRESULT_FROM_WIN32(containment);
  HRESULT status = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
  if (FAILED(status)) return status;
  ComApartment apartment;
  status = CoInitializeSecurity(nullptr, -1, nullptr, nullptr, RPC_C_AUTHN_LEVEL_PKT_PRIVACY,
    RPC_C_IMP_LEVEL_IMPERSONATE, nullptr, EOAC_NONE, nullptr);
  if (FAILED(status)) return status;
  ComRef<ITaskService> service;
  status = CoCreateInstance(CLSID_TaskScheduler, nullptr, CLSCTX_INPROC_SERVER,
    IID_ITaskService, reinterpret_cast<void **>(&service.value));
  if (FAILED(status)) return status;
  VARIANT empty; VariantInit(&empty);
  status = service.value->Connect(empty, empty, empty, empty);
  if (FAILED(status)) return status;
  Bstr root; root.value = SysAllocString(L"\\");
  Bstr name; name.value = SysAllocString(L"MagnitudeInference");
  if (!root.value || !name.value) return E_OUTOFMEMORY;
  ComRef<ITaskFolder> folder;
  status = service.value->GetFolder(root.value, &folder.value);
  if (FAILED(status)) return status;
  ComRef<IRegisteredTask> task;
  status = folder.value->GetTask(name.value, &task.value);
  if (status == HRESULT_FROM_WIN32(ERROR_FILE_NOT_FOUND)) {
    output = "{\"_tag\":\"Missing\"}";
    return S_OK;
  }
  if (FAILED(status)) return status;
  Bstr xml;
  status = task.value->get_Xml(&xml.value);
  if (FAILED(status)) return status;
  std::string document, sid;
  status = utf8(xml.value, SysStringLen(xml.value), document);
  if (FAILED(status)) return status;
  status = current_sid(sid);
  if (FAILED(status)) return status;
  if (expected_digest) {
    std::string captured_sid;
    status = utf8(expected_sid, static_cast<UINT>(wcslen(expected_sid)), captured_sid);
    if (FAILED(status)) return status;
    if (captured_sid != sid) return E_ACCESSDENIED;
    status = digest_matches(document, expected_digest);
    if (FAILED(status)) return status;
    status = folder.value->DeleteTask(name.value, 0);
    if (FAILED(status) && status != HRESULT_FROM_WIN32(ERROR_FILE_NOT_FOUND)) return status;
    ComRef<IRegisteredTask> remaining;
    status = folder.value->GetTask(name.value, &remaining.value);
    if (status != HRESULT_FROM_WIN32(ERROR_FILE_NOT_FOUND))
      return FAILED(status) ? status : HRESULT_FROM_WIN32(ERROR_REVISION_MISMATCH);
    output = "{\"_tag\":\"Missing\"}";
    return S_OK;
  }
  // Enablement is parsed from this same XML snapshot, never a second racy read.
  output = "{\"_tag\":\"Registered\",\"xml\":" + json(document) + ",\"currentUserSid\":" + json(sid) + "}";
  return S_OK;
}
int wmain(int argc, WCHAR **argv) {
  HRESULT status = E_INVALIDARG;
  std::string output;
  try {
    if (argc == 1) status = query(output, nullptr, nullptr);
    else if (argc == 4 && wcscmp(argv[1], L"--retire") == 0)
      status = query(output, argv[2], argv[3]);
  }
  catch (const std::bad_alloc &) { status = E_OUTOFMEMORY; }
  if (FAILED(status)) {
    std::printf("{\"_tag\":\"Failed\",\"hresult\":%lu}\n", static_cast<unsigned long>(static_cast<DWORD>(status)));
    return 1;
  }
  if (std::fwrite(output.data(), 1, output.size(), stdout) != output.size() || std::fputc('\n', stdout) == EOF || std::fflush(stdout)) return 2;
  return 0;
}
