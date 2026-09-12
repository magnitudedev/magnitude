/* Local device description only. No serial number or UUID crosses this boundary. */
#include <node_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#ifdef _WIN32
static int smbios_string(const unsigned char *strings, const unsigned char *end, unsigned index, char *out) {
  if (!index) return 0;
  while (strings < end && *strings) {
    const unsigned char *nul = memchr(strings, 0, (size_t)(end - strings));
    if (!nul) return 0;
    if (--index == 0) {
      size_t length = (size_t)(nul - strings);
      if (!length || length > 255) return 0;
      memcpy(out, strings, length); out[length] = 0; return 1;
    }
    strings = nul + 1;
  }
  return 0;
}
static int read_identity(char *manufacturer, char *model) {
  const DWORD provider = 0x52534d42; /* RSMB */
  UINT size = GetSystemFirmwareTable(provider, 0, NULL, 0);
  if (size < 8 || size > 1024 * 1024) return 0;
  unsigned char *data = malloc(size); if (!data) return 0;
  int ok = 0;
  if (GetSystemFirmwareTable(provider, 0, data, size) != size) goto done;
  DWORD length; memcpy(&length, data + 4, 4);
  if (length > size - 8) goto done;
  const unsigned char *entry = data + 8, *end = entry + length;
  while (end - entry >= 4) {
    unsigned formatted = entry[1];
    if (formatted < 4 || formatted > (size_t)(end - entry)) break;
    const unsigned char *strings = entry + formatted, *next = strings;
    while (end - next >= 2 && (next[0] || next[1])) next++;
    if (end - next < 2) break;
    if (entry[0] == 1 && formatted >= 6) {
      ok = smbios_string(strings, next + 1, entry[4], manufacturer) && smbios_string(strings, next + 1, entry[5], model);
      break;
    }
    entry = next + 2;
  }
done:
  free(data); return ok;
}
#elif defined(__APPLE__)
static int read_identity(char *manufacturer, char *model) {
  size_t size = 256;
  if (sysctlbyname("hw.model", model, &size, NULL, 0) || !size || size > 256) return 0;
  model[255] = 0; strcpy(manufacturer, "Apple"); return 1;
}
#else
static int read_text(const char *path, char *out) {
  FILE *file = fopen(path, "r"); if (!file) return 0;
  size_t size = fread(out, 1, 255, file);
  int ok = !ferror(file) && size > 0 && size < 255;
  fclose(file); out[size] = 0; return ok;
}
static int read_identity(char *manufacturer, char *model) {
  return read_text("/sys/devices/virtual/dmi/id/sys_vendor", manufacturer) &&
    read_text("/sys/devices/virtual/dmi/id/product_name", model);
}
#endif

static napi_value observe(napi_env env, napi_callback_info info) {
  (void)info; char manufacturer[256] = {0}, model[256] = {0};
  if (!read_identity(manufacturer, model)) { napi_throw_error(env, NULL, "Device identity unavailable"); return NULL; }
  napi_value result, value; napi_create_object(env, &result);
  napi_create_string_utf8(env, manufacturer, NAPI_AUTO_LENGTH, &value); napi_set_named_property(env, result, "manufacturer", value);
  napi_create_string_utf8(env, model, NAPI_AUTO_LENGTH, &value); napi_set_named_property(env, result, "model", value);
  return result;
}
void magnitude_register_machine_identity(napi_env env, napi_value exports) {
  napi_property_descriptor method = {"machineIdentity", NULL, observe, NULL, NULL, NULL, napi_default, NULL};
  napi_define_properties(env, exports, 1, &method);
}
