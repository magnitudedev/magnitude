#include "mac-update-validation.h"
#include <CoreFoundation/CoreFoundation.h>
#include <Security/Security.h>
#include <fcntl.h>
#include <limits.h>
#include <mach-o/fat.h>
#include <mach-o/loader.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int string_matches(CFTypeRef value, CFStringRef expected) {
  return value && expected && CFGetTypeID(value) == CFStringGetTypeID() &&
    CFGetTypeID(expected) == CFStringGetTypeID() && CFEqual(value, expected);
}

static int is_mach_executable(CFDictionaryRef information) {
  CFTypeRef executable = CFDictionaryGetValue(information, kSecCodeInfoMainExecutable);
  UInt8 path[PATH_MAX];
  if (!executable || CFGetTypeID(executable) != CFURLGetTypeID() ||
      !CFURLGetFileSystemRepresentation(executable, true, path, sizeof(path))) return 0;
  int fd = open((const char *)path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
  if (fd < 0) return 0;
  struct stat st;
  uint32_t magic = 0;
  int valid = fstat(fd, &st) == 0 && S_ISREG(st.st_mode) &&
    read(fd, &magic, sizeof(magic)) == sizeof(magic) &&
    (magic == MH_MAGIC_64 || magic == MH_CIGAM_64 || magic == FAT_MAGIC ||
     magic == FAT_CIGAM || magic == FAT_MAGIC_64 || magic == FAT_CIGAM_64);
  close(fd);
  return valid;
}

int magnitude_verify_mac_bundle(const char *path, const char *requirement,
    const char *version, const char *architecture) {
  int valid = 0, directory = -1;
  CFURLRef url = NULL;
  CFStringRef rule = NULL, expected_version = NULL, arch = NULL;
  CFDictionaryRef attributes = NULL, information = NULL;
  SecRequirementRef policy = NULL;
  SecStaticCodeRef code = NULL, all_architectures = NULL;
  char resolved[PATH_MAX];
  struct stat retained, before, after;
  if (!path || path[0] != '/' || !requirement || !*requirement || !version || !*version ||
      !architecture || (strcmp(architecture, "arm64") && strcmp(architecture, "x86_64"))) goto done;
  /* Reject the supplied leaf symlink before canonicalizing ancestors for Security.framework. */
  if (lstat(path, &before) || !S_ISDIR(before.st_mode) || !realpath(path, resolved)) goto done;
  directory = open(resolved, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
  if (directory < 0 || fstat(directory, &retained) ||
      retained.st_dev != before.st_dev || retained.st_ino != before.st_ino) goto done;
  url = CFURLCreateFromFileSystemRepresentation(NULL, (const UInt8 *)resolved, strlen(resolved), true);
  rule = CFStringCreateWithCString(NULL, requirement, kCFStringEncodingUTF8);
  expected_version = CFStringCreateWithCString(NULL, version, kCFStringEncodingUTF8);
  arch = CFStringCreateWithCString(NULL, architecture, kCFStringEncodingUTF8);
  if (!url || !rule || !expected_version || !arch) goto done;
  const void *keys[] = {kSecCodeAttributeArchitecture};
  const void *values[] = {arch};
  attributes = CFDictionaryCreate(NULL, keys, values, 1, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
  /* Selecting a slice at creation narrows validation even with CheckAllArchitectures.
     Validate an unqualified object first, then require the intended executable slice. */
  if (!attributes || SecRequirementCreateWithString(rule, kSecCSDefaultFlags, &policy) != errSecSuccess ||
      SecStaticCodeCreateWithPath(url, kSecCSDefaultFlags, &all_architectures) != errSecSuccess ||
      SecStaticCodeCheckValidity(all_architectures,
        kSecCSCheckAllArchitectures | kSecCSCheckNestedCode | kSecCSStrictValidate, policy) != errSecSuccess ||
      SecStaticCodeCreateWithPathAndAttributes(url, kSecCSDefaultFlags, attributes, &code) != errSecSuccess ||
      SecCodeCopySigningInformation(code, kSecCSDefaultFlags, &information) != errSecSuccess) goto done;
  CFTypeRef plist = CFDictionaryGetValue(information, kSecCodeInfoPList);
  if (!plist || CFGetTypeID(plist) != CFDictionaryGetTypeID() ||
      !string_matches(CFDictionaryGetValue(plist, CFSTR("CFBundleIdentifier")),
        CFDictionaryGetValue(information, kSecCodeInfoIdentifier)) ||
      !string_matches(CFDictionaryGetValue(plist, CFSTR("CFBundleShortVersionString")), expected_version) ||
      !string_matches(CFDictionaryGetValue(plist, CFSTR("CFBundlePackageType")), CFSTR("APPL")) ||
      !is_mach_executable(information)) goto done;
  if (lstat(resolved, &after) || !S_ISDIR(after.st_mode) ||
      retained.st_dev != after.st_dev || retained.st_ino != after.st_ino ||
      lstat(path, &after) || !S_ISDIR(after.st_mode) ||
      retained.st_dev != after.st_dev || retained.st_ino != after.st_ino) goto done;
  valid = 1;
done:
  if (information) CFRelease(information);
  if (code) CFRelease(code);
  if (all_architectures) CFRelease(all_architectures);
  if (policy) CFRelease(policy);
  if (attributes) CFRelease(attributes);
  if (arch) CFRelease(arch);
  if (expected_version) CFRelease(expected_version);
  if (rule) CFRelease(rule);
  if (url) CFRelease(url);
  if (directory >= 0) close(directory);
  return valid;
}
