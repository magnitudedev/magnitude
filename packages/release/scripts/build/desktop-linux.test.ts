import { BunContext } from "@effect/platform-bun"
import { Effect, Option } from "effect"
import { spawnSync } from "node:child_process"
import { chmodSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { gunzipSync } from "node:zlib"
import { describe, expect, it } from "vitest"
import { buildLinuxDesktopInstaller, mtreeFileModes, validateLinuxPayloadPermissions, validateLinuxSystemFiles, validateLinuxUpdatePolicy, validatePacmanMetadataModes } from "./desktop-linux"
import { linuxSystemFiles } from "./linux-system-files"
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

describe("Linux system files", () => {
  const listing = new Map<string, { mode: number; owner: string }>(linuxSystemFiles.map(file => [file.path, { mode: file.mode, owner: "root/root" }]))
  const validate = (entries: Map<string, { mode: number; owner: string }>) =>
    Effect.runPromise(validateLinuxSystemFiles(path => Option.fromNullable(entries.get(path))))
  it("accepts the root-owned unit, Polkit action, installer wrapper and 0440 sudoers rule", async () => {
    await expect(validate(listing)).resolves.toBeUndefined()
  })
  it.each([
    ["a group-writable installer wrapper", "usr/lib/magnitude-desktop/resources/magnitude-install-update", { mode: 0o775, owner: "root/root" }],
    ["a setuid installer wrapper", "usr/lib/magnitude-desktop/resources/magnitude-install-update", { mode: 0o4755, owner: "root/root" }],
    ["a foreign-owned installer wrapper", "usr/lib/magnitude-desktop/resources/magnitude-install-update", { mode: 0o755, owner: "builder/root" }],
    ["a group-readable sudoers rule", "etc/sudoers.d/magnitude", { mode: 0o640, owner: "root/root" }],
    ["a writable sudoers rule", "etc/sudoers.d/magnitude", { mode: 0o644, owner: "root/root" }],
    ["a foreign-owned sudoers rule", "etc/sudoers.d/magnitude", { mode: 0o440, owner: "builder/root" }],
    ["a group-writable unit", "usr/lib/systemd/system/magnitude.service", { mode: 0o664, owner: "root/root" }],
  ] as const)("rejects %s", async (_, path, entry) => {
    await expect(validate(new Map([...listing, [path, entry]]))).rejects.toThrow("root-owned")
  })
  it.each(["polkit", "magnitude-install-update"])("rejects a package missing %s", async name => {
    await expect(validate(new Map([...listing].filter(([path]) => !path.includes(name))))).rejects.toThrow("root-owned")
  })
  it("passes visudo", () => {
    const visudo = ["/usr/sbin/visudo", "/usr/bin/visudo"].find(path => spawnSync(path, ["-V"]).status === 0)
    if (!visudo) return
    expect(spawnSync(visudo, ["-cf", linuxSystemFiles.find(file => file.path.startsWith("etc/"))!.source]).status).toBe(0)
  })
})

describe("Linux update Polkit action", () => {
  const source = (name: string) => linuxSystemFiles.find(file => file.path.endsWith(name))!.source
  const policy = readFileSync(source(".policy"), "utf8")
  const annotation = (key: string, value: string) => `<annotate key="org.freedesktop.policykit.exec.${key}">${value}</annotate>`
  const wrapperPath = "/usr/lib/magnitude-desktop/resources/magnitude-install-update"
  it("names only the installer wrapper for password-free authorization", async () => {
    expect(policy).toContain("<allow_active>yes</allow_active>")
    await expect(Effect.runPromise(validateLinuxUpdatePolicy(policy))).resolves.toBeUndefined()
  })
  it.each([
    ["the CLI itself", policy.replace(wrapperPath, "/usr/lib/magnitude-desktop/resources/magnitude")],
    ["an argument annotation polkit 0.105 ignores", policy.replace("</annotate>", `</annotate>\n    ${annotation("argv1", "_install-application-update")}`)],
    ["no program", policy.replace(/\s*<annotate[^\n]*\n/, "\n")],
  ])("rejects an action naming %s", async (_, candidate) => {
    await expect(Effect.runPromise(validateLinuxUpdatePolicy(candidate))).rejects.toThrow("must name only")
  })

  describe("installer wrapper", () => {
    const root = mkdtempSync(join(tmpdir(), "magnitude-wrapper-"))
    const cli = join(root, "magnitude")
    const wrapper = join(root, "magnitude-install-update")
    writeFileSync(cli, "#!/bin/sh\nprintf '%s\\n' \"$@\"\n")
    chmodSync(cli, 0o755)
    const text = readFileSync(source("magnitude-install-update"), "utf8")
    writeFileSync(wrapper, text.replace("exec /usr/lib/magnitude-desktop/resources/magnitude ", `exec ${cli} `))
    const run = (...args: string[]) => {
      const result = spawnSync("/bin/sh", [wrapper, ...args], { env: {} })
      return { status: result.status, stdout: result.stdout.toString() }
    }
    it("runs the installed CLI by absolute path", () => {
      expect(text.startsWith("#!/bin/sh\n")).toBe(true)
      expect(text).toMatch(/^set -eu$/m)
      expect(text.match(/^exec .*$/gm)).toEqual(['exec /usr/lib/magnitude-desktop/resources/magnitude _install-application-update "$@"'])
    })
    it.each([
      [["/home/ada/.magnitude/updates/update.json"]],
      [["/home/ada/.magnitude/updates/update.json", "--parent-stdin"]],
    ])("passes %j to the installer entry", args => {
      expect(run(...args)).toEqual({ status: 0, stdout: `${["_install-application-update", ...args].join("\n")}\n` })
    })
    it.each([
      [[]],
      [["updates/update.json"]],
      [["--help"]],
      [["_server-remove"]],
      [["serve", "/a"]],
      [["/a/updates/update.json", "--force"]],
      [["/a/updates/update.json", "--parent-stdin", "x"]],
      [["--parent-stdin", "/a/updates/update.json"]],
      [["-c", "id"]],
    ])("refuses %j without running the CLI", args => {
      expect(run(...args)).toEqual({ status: 3, stdout: "" })
    })
  })
})

describe("pacman package metadata modes", () => {
  const protectedEntries = new Map([".PKGINFO", ".INSTALL", ".MTREE"].map(path => [path, "-rw-r--r-- root root"] as const))
  const manifest = (mode: string) => `#mtree\n/set type=file uid=0 gid=0 mode=${mode}\n./.INSTALL time=1 size=1145\n./.PKGINFO time=1 size=830\n/set mode=755\n./usr type=dir\n`

  it("resolves modes from /set defaults and entry overrides", () => {
    const modes = mtreeFileModes(`${manifest("644")}./usr/lib/magnitude-desktop/chrome-sandbox mode=4755\n`)
    expect(modes.get(".INSTALL")).toBe(0o644)
    expect(modes.get("usr")).toBe(0o755)
    expect(modes.get("usr/lib/magnitude-desktop/chrome-sandbox")).toBe(0o4755)
  })

  it("accepts makepkg-equivalent metadata", async () => {
    await expect(Effect.runPromise(validatePacmanMetadataModes(protectedEntries, manifest("644")))).resolves.toBeUndefined()
  })

  it("rejects metadata the manifest records as group-writable", async () => {
    await expect(Effect.runPromise(validatePacmanMetadataModes(protectedEntries, manifest("664")))).rejects.toThrow("mode 0644")
  })

  it.each([".PKGINFO", ".INSTALL", ".MTREE"])("rejects a group-writable %s archive entry", async path => {
    const entries = new Map<string, string>(protectedEntries).set(path, "-rw-rw-r-- root root")
    await expect(Effect.runPromise(validatePacmanMetadataModes(entries, manifest("644")))).rejects.toThrow("mode 0644")
  })
})

const hasBsdtar = spawnSync("/bin/sh", ["-c", "command -v bsdtar"]).status === 0
describe.skipIf(process.platform !== "linux" || !hasBsdtar)("pacman package assembly", () => {
  it("produces a root-owned package with admission hook, scriptlets and setuid sandbox", async () => {
    const root = mkdtempSync(join(tmpdir(), "magnitude-pacman-"))
    // Release build hosts use umask 002; metadata must not inherit its group write bit.
    const umask = process.umask(0o002)
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
      expect(mtree).toMatch(/^\.\/etc\/sudoers\.d\/magnitude .*mode=440/m)
      expect(mtree).toMatch(/^\.\/etc\/sudoers\.d .*mode=750/m)
      expect(read("etc/sudoers.d/magnitude").toString()).toContain("magnitude ALL=(root) NOPASSWD: /usr/lib/magnitude-desktop/resources/magnitude _install-application-update *")
      expect(read("usr/lib/systemd/system/magnitude.service").toString()).toContain("ExecStart=/usr/bin/env /usr/lib/magnitude-desktop/resources/magnitude serve")
      expect(read("usr/share/polkit-1/actions/dev.magnitude.update.policy").toString()).toContain("<allow_active>yes</allow_active>")
      expect(read("usr/lib/magnitude-desktop/resources/magnitude-install-update").toString()).toContain("_install-application-update")
      expect(mtreeFileModes(mtree).get("usr/lib/magnitude-desktop/resources/magnitude-install-update")).toBe(0o755)
      expect(mtreeFileModes(mtree).get(".INSTALL")).toBe(0o644)
      expect(mtreeFileModes(mtree).get(".PKGINFO")).toBe(0o644)
      const listing = spawnSync("bsdtar", ["-tvf", output], { env: { ...process.env, LC_ALL: "C" } }).stdout.toString()
      for (const member of [".MTREE", ".PKGINFO", ".INSTALL"]) {
        expect(listing).toMatch(new RegExp(`^-rw-r--r-- +0 root +root .* \\${member}$`, "m"))
      }
    } finally {
      process.umask(umask)
      rmSync(root, { recursive: true, force: true })
    }
  })
})
