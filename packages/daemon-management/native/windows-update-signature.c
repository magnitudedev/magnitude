#include "windows-update-signature.h"
#include <wincrypt.h>
#include <wintrust.h>
#include <softpub.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

static BOOL matches_organization(PCCERT_CONTEXT certificate, const WCHAR *organization) {
  CERT_NAME_INFO *name = NULL;
  DWORD size = 0;
  if (!CryptDecodeObjectEx(X509_ASN_ENCODING, X509_NAME,
      certificate->pCertInfo->Subject.pbData, certificate->pCertInfo->Subject.cbData,
      CRYPT_DECODE_ALLOC_FLAG, NULL, &name, &size)) return FALSE;
  BOOL matched = FALSE;
  for (DWORD rdn = 0; rdn < name->cRDN && !matched; ++rdn) {
    for (DWORD index = 0; index < name->rgRDN[rdn].cRDNAttr && !matched; ++index) {
      CERT_RDN_ATTR *attribute = &name->rgRDN[rdn].rgRDNAttr[index];
      if (strcmp(attribute->pszObjId, szOID_ORGANIZATION_NAME)) continue;
      WCHAR value[1024];
      DWORD length = CertRDNValueToStrW(attribute->dwValueType, &attribute->Value, value, 1024);
      matched = length > 1 && length <= 1024 && !wcscmp(value, organization);
    }
  }
  LocalFree(name); return matched;
}

DWORD magnitude_verify_installer_signature(const WCHAR *path, const WCHAR *organization) {
  if (!path || !path[0] || !organization || !organization[0] || wcslen(organization) >= 1024)
    return ERROR_INVALID_PARAMETER;
  HANDLE file = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
    FILE_FLAG_OPEN_REPARSE_POINT, NULL);
  if (file == INVALID_HANDLE_VALUE) return GetLastError();
  BY_HANDLE_FILE_INFORMATION info;
  if (GetFileType(file) != FILE_TYPE_DISK || !GetFileInformationByHandle(file, &info) ||
      info.nNumberOfLinks != 1 || (info.dwFileAttributes & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT))) {
    CloseHandle(file); return ERROR_ACCESS_DENIED;
  }
  WINTRUST_FILE_INFO input = {0};
  input.cbStruct = sizeof(input); input.pcwszFilePath = path; input.hFile = file;
  WINTRUST_DATA trust = {0};
  trust.cbStruct = sizeof(trust);
  trust.dwUIChoice = WTD_UI_NONE;
  trust.fdwRevocationChecks = WTD_REVOKE_WHOLECHAIN;
  trust.dwUnionChoice = WTD_CHOICE_FILE;
  trust.pFile = &input;
  trust.dwStateAction = WTD_STATEACTION_VERIFY;
  trust.dwUIContext = WTD_UICONTEXT_INSTALL;
  GUID action = WINTRUST_ACTION_GENERIC_VERIFY_V2;
  DWORD error = (DWORD)WinVerifyTrust(INVALID_HANDLE_VALUE, &action, &trust);
  if (!error) {
    CRYPT_PROVIDER_DATA *provider = WTHelperProvDataFromStateData(trust.hWVTStateData);
    CRYPT_PROVIDER_SGNR *signer = provider ? WTHelperGetProvSignerFromChain(provider, 0, FALSE, 0) : NULL;
    if (!signer || !signer->csCertChain || !signer->pasCertChain[0].pCert ||
        !matches_organization(signer->pasCertChain[0].pCert, organization)) error = ERROR_ACCESS_DENIED;
  }
  trust.dwStateAction = WTD_STATEACTION_CLOSE;
  DWORD closed = (DWORD)WinVerifyTrust(INVALID_HANDLE_VALUE, &action, &trust);
  CloseHandle(file);
  return error ? error : closed;
}
