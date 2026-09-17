/* Local device description only. No serial number or UUID crosses this boundary. */
#include <node_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "machine-identity-data.h"
#ifdef _WIN32
#include <windows.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#ifdef _WIN32
static void read_identity(magnitude_machine_identity *out) {
  const DWORD provider = 0x52534d42; /* RSMB */
  UINT size = GetSystemFirmwareTable(provider, 0, NULL, 0);
  if (size < 8 || size > 1024 * 1024) return;
  unsigned char *data = malloc(size); if (!data) return;
  if (GetSystemFirmwareTable(provider, 0, data, size) == size) {
    DWORD length; memcpy(&length, data + 4, 4);
    if (length <= size - 8) magnitude_parse_smbios(data + 8, length, out);
  }
  free(data);
}
#elif defined(__APPLE__)
static void read_identity(magnitude_machine_identity *out) {
  size_t size = sizeof(out->model);
  if (sysctlbyname("hw.model", out->model, &size, NULL, 0) || !size || size > sizeof(out->model)) { out->model[0] = 0; return; }
  out->model[255] = 0; strcpy(out->manufacturer, "Apple");
  /* Modern Mac identifiers are opaque; curated enclosure matches classify them. */
}
#else
static void read_text(const char *path, char *out) {
  FILE *file = fopen(path, "r"); if (!file) return;
  size_t size = fread(out, 1, 255, file);
  int ok = !ferror(file) && size > 0 && size < 255;
  fclose(file); out[ok ? size : 0] = 0;
}
static void read_identity(magnitude_machine_identity *out) {
  read_text("/sys/devices/virtual/dmi/id/sys_vendor", out->manufacturer);
  read_text("/sys/devices/virtual/dmi/id/product_name", out->model);
  read_text("/sys/devices/virtual/dmi/id/product_family", out->family);
  read_text("/sys/devices/virtual/dmi/id/product_version", out->version);
  char chassis[256] = {0}; read_text("/sys/devices/virtual/dmi/id/chassis_type", chassis);
  char *end; unsigned long type = strtoul(chassis, &end, 10);
  while (*end == ' ' || *end == '\n' || *end == '\r') end++;
  if (end != chassis && !*end && type <= 127) out->chassis_type = (unsigned)type;
}
#endif

static napi_value observe(napi_env env, napi_callback_info info) {
  (void)info; magnitude_machine_identity identity = {0}; read_identity(&identity);
  napi_value result, value; napi_create_object(env, &result);
  const char *keys[] = {"manufacturer", "model", "family", "version"};
  const char *values[] = {identity.manufacturer, identity.model, identity.family, identity.version};
  for (unsigned i = 0; i < 4; i++) {
    napi_create_string_utf8(env, values[i], NAPI_AUTO_LENGTH, &value); napi_set_named_property(env, result, keys[i], value);
  }
  napi_create_uint32(env, identity.chassis_type, &value); napi_set_named_property(env, result, "chassisType", value);
  return result;
}
void magnitude_register_machine_identity(napi_env env, napi_value exports) {
  napi_property_descriptor method = {"machineIdentity", NULL, observe, NULL, NULL, NULL, napi_default, NULL};
  napi_define_properties(env, exports, 1, &method);
}
