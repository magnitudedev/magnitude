#ifndef MAGNITUDE_MACHINE_IDENTITY_DATA_H
#define MAGNITUDE_MACHINE_IDENTITY_DATA_H
#include <stddef.h>
#include <string.h>

typedef struct {
  char manufacturer[256], model[256], family[256], version[256];
  unsigned chassis_type;
} magnitude_machine_identity;

static inline int magnitude_smbios_string(const unsigned char *strings, const unsigned char *end, unsigned index, char *out) {
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
/* Parse the bounded SMBIOS structure table (without the Windows eight-byte header).
 * Type 1 supplies product identity; Type 3 supplies enclosure kind, in either order.
 * Never read serial, UUID, asset tag, or unrelated device strings. */
static inline void magnitude_parse_smbios(const unsigned char *entry, size_t length, magnitude_machine_identity *out) {
  const unsigned char *end = entry + length;
  int system_seen = 0, chassis_seen = 0;
  while (end - entry >= 4) {
    unsigned formatted = entry[1];
    if (formatted < 4 || formatted > (size_t)(end - entry)) break;
    const unsigned char *strings = entry + formatted, *next = strings;
    while (end - next >= 2 && (next[0] || next[1])) next++;
    if (end - next < 2) break;
    if (entry[0] == 127) break;
    if (entry[0] == 1 && formatted >= 6 && !system_seen) {
      system_seen = 1;
      magnitude_smbios_string(strings, next + 1, entry[4], out->manufacturer);
      magnitude_smbios_string(strings, next + 1, entry[5], out->model);
      if (formatted >= 7) magnitude_smbios_string(strings, next + 1, entry[6], out->version);
      if (formatted >= 27) magnitude_smbios_string(strings, next + 1, entry[26], out->family);
    }
    if (entry[0] == 3 && formatted >= 6 && !chassis_seen) {
      chassis_seen = 1;
      out->chassis_type = entry[5] & 0x7f;
    }
    entry = next + 2;
  }
}
#endif
