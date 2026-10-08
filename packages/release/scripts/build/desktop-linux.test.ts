import { BunContext } from "@effect/platform-bun"
import { Effect } from "effect"
import { spawnSync } from "node:child_process"
import { chmodSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { gunzipSync } from "node:zlib"
import { describe, expect, it } from "vitest"
import { buildLinuxDesktopInstaller, validateLinuxPayloadPermissions } from "./desktop-linux"
import { linuxDesktopInstaller } from "../../src/targets"

describe("Linux prerelease asset names", () => {
  it.each([
    ["linux-arm64-gnu", "deb", "magnitude-desktop_0.1.0-alpha.0-40_arm64.deb"],
    ["linux-x64-gnu", "deb", "magnitude-desktop_0.1.0-alpha.0-40_amd64.deb"],
    ["linux-arm64-gnu", "rpm", "magnitude-desktop-0.1.0-alpha.0-40.aarch64.rpm"],
    ["linux-x64-gnu", "rpm", "magnitude-desktop-0.1.0-alpha.0-40.x86_64.rpm"],
    ["linux-x64-gnu", "pacman", "magnitude-desktop-0.1.0alpha.0-40-x86_64.pkg.tar.zst"],
  ] as const)("names the %s %s asset", (host, format, expected) => {
    expect(linuxDesktopInstaller(host, format, "0.1.0-alpha.0", 40)).toBe(expected)
  })
})

describe("Linux package publisher-trust protection", () => {
  it.each([
    ["deb", "drwxr-xr-x root/root 0 2026-09-13 00:00 ./usr/lib/magnitude-desktop/resources/\n-rw-r--r-- root/root 153 2026-09-13 00:00 ./usr/lib/magnitude-desktop/resources/update-trust.json"],
    ["rpm", "/usr/lib/magnitude-desktop/resources 40755 root root\n/usr/lib/magnitude-desktop/resources/update-trust.json 100644 root root"],
    ["pacman", "drwxr-xr-x  0 root   root        0 Oct  7 18:31 usr/lib/magnitude-desktop/resources/\n-rw-r--r--  0 root   root      153 Oct  7 18:31 usr/lib/magnitude-desktop/resources/update-trust.json"],
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
    ["pacman", "drwxrwxr-x  0 root   root        0 Oct  7 18:31 usr/lib/magnitude-desktop/resources/"],
    ["pacman", "-rw-r--rw-  0 root   root      153 Oct  7 18:31 usr/lib/magnitude-desktop/resources/update-trust.json"],
    ["pacman", "-rw-r--r--  0 builder root      153 Oct  7 18:31 usr/lib/magnitude-desktop/resources/update-trust.json"],
    ["deb", ""], ["rpm", ""], ["pacman", ""],
  ] as const)("rejects unprotected or missing %s payload: %s", async (format, listing) => {
    await expect(Effect.runPromise(validateLinuxPayloadPermissions(format, listing))).rejects.toThrow("root-owned")
  })
})

const hasBsdtar = spawnSync("/bin/sh", ["-c", "command -v bsdtar"]).status === 0
describe.skipIf(process.platform !== "linux" || !hasBsdtar)("pacman package assembly", () => {
  it("produces a root-owned package with admission hook, scriptlets and setuid sandbox", async () => {
    const root = mkdtempSync(join(tmpdir(), "magnitude-pacman-"))
    try {
      const app = join(root, "app")
      mkdirSync(join(app, "resources"), { recursive: true })
      for (const [path, mode] of [["magnitude", 0o775], ["chrome-sandbox", 0o775], ["resources/magnitude", 0o775], ["resources/update-trust.json", 0o664]] as const) {
        writeFileSync(join(app, path), path)
        chmodSync(join(app, path), mode)
      }
      const { output, artifact } = await Effect.runPromise(buildLinuxDesktopInstaller({ format: "pacman", arch: "x64", version: "1.2.0-beta.3", revision: 7, app, output: join(root, "out") })
        .pipe(Effect.provide(BunContext.layer)))
      expect(artifact).toMatchObject({ id: "desktop-linux-x64-gnu-pacman", filename: "magnitude-desktop-1.2.0beta.3-7-x86_64.pkg.tar.zst" })
      const read = (member: string) => spawnSync("bsdtar", ["-xOf", output, member]).stdout as Buffer
      expect(read(".PKGINFO").toString()).toMatch(/^pkgver = 1\.2\.0beta\.3-7$/m)
      expect(read("usr/lib/magnitude-desktop/resources/update-package.json").toString()).toBe('{"format":"pacman"}')
      const install = join(root, "install.sh")
      writeFileSync(install, read(".INSTALL"))
      expect(spawnSync("/bin/sh", ["-n", install]).status).toBe(0)
      const mtree = gunzipSync(read(".MTREE")).toString()
      expect(mtree).toMatch(/uid=0 gid=0/)
      expect(mtree).not.toMatch(/uid=[1-9]/)
      expect(mtree).toMatch(/^\.\/usr\/lib\/magnitude-desktop\/chrome-sandbox .*mode=4755/m)
    } finally { rmSync(root, { recursive: true, force: true }) }
  })
})
