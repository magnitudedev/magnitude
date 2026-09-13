import { Effect } from "effect"
import { describe, expect, it } from "vitest"
import { validateLinuxPayloadPermissions } from "./desktop-linux"

describe("Linux package publisher-trust protection", () => {
  it.each([
    ["deb", "drwxr-xr-x root/root 0 2026-09-13 00:00 ./usr/lib/magnitude-desktop/resources/\n-rw-r--r-- root/root 153 2026-09-13 00:00 ./usr/lib/magnitude-desktop/resources/update-trust.json"],
    ["rpm", "/usr/lib/magnitude-desktop/resources 40755 root root\n/usr/lib/magnitude-desktop/resources/update-trust.json 100644 root root"],
  ] as const)("accepts protected %s package contents", async (format, listing) => {
    await expect(Effect.runPromise(validateLinuxPayloadPermissions(format, listing))).resolves.toBeUndefined()
  })
  it.each([
    ["deb", "drwxrwxr-x root/root 0 2026-09-13 00:00 ./usr/lib/magnitude-desktop/resources/"],
    ["deb", "-rw-r--rw- root/root 153 2026-09-13 00:00 ./usr/lib/magnitude-desktop/resources/update-trust.json"],
    ["deb", "-rw-r--r-- builder/root 153 2026-09-13 00:00 ./usr/lib/magnitude-desktop/resources/update-trust.json"],
    ["rpm", "/usr/lib/magnitude-desktop/resources 40775 root root"],
    ["rpm", "/usr/lib/magnitude-desktop 40775 root root"],
    ["rpm", "/usr/lib/magnitude-desktop/resources/update-trust.json 100646 root root"],
    ["rpm", "/usr/lib/magnitude-desktop/resources/update-trust.json 100644 builder root"],
    ["deb", ""], ["rpm", ""],
  ] as const)("rejects unprotected or missing %s payload: %s", async (format, listing) => {
    await expect(Effect.runPromise(validateLinuxPayloadPermissions(format, listing))).rejects.toThrow("root-owned")
  })
})
