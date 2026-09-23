#ifndef MAGNITUDE_MAC_UPDATE_VALIDATION_H
#define MAGNITUDE_MAC_UPDATE_VALIDATION_H

/* Caller retains exclusive staging ownership throughout verification and publication. */
int magnitude_verify_mac_bundle(const char *path, const char *requirement,
  const char *version, const char *architecture);

#endif
