import { NodeContext } from "@effect/platform-node"
import { Effect } from "effect"
import { createHash, generateKeyPairSync } from "node:crypto"
import { mkdtempSync, writeFileSync, mkdirSync, readFileSync, existsSync, rmSync, symlinkSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { spawnSync } from "node:child_process"
import { describe, expect, it } from "vitest"
import { signUpdateRelease } from "../../src/hosted-update/release"
import { renderUnixInstallationScript } from "./installation-scripts"

// The script selects its package manager by availability, so each case sees only its own.
const setsid = spawnSync("/bin/sh", ["-c", "command -v setsid"], { encoding: "utf8" }).stdout.trim()
const systemTools = ["cat", "cp", "cut", "id", "kill", "mkdir", "mktemp", "openssl", "python3", "rm", "sha256sum", "sh", "sleep", "timeout", "tr", "uname", "wc"]
const managers = [
  { manager: "apt-get", package: "deb", receipt: /^install\n-y\n.*magnitude\.deb\n$/ },
  { manager: "pacman", package: "pacman", receipt: /^-U\n--noconfirm\n.*magnitude\.pacman\n$/ },
] as const

describe.skipIf(process.platform !== "linux").each(managers)("Linux installation shell verification with $manager", ({ manager, package: format, receipt: expected }) => {
  it.each(["valid", "signature", "digest", "architecture", "channel"])("admits the package manager only for %s input", async scenario => {
    if (format === "pacman" && process.arch !== "x64") return
    const root = mkdtempSync(join(tmpdir(), "magnitude shell ' "))
    try {
      const bin = join(root, "bin"), tools = join(root, "tools")
      mkdirSync(bin)
      mkdirSync(tools)
      for (const tool of systemTools) {
        const found = spawnSync("/bin/sh", ["-c", `command -v ${tool}`], { encoding: "utf8" }).stdout.trim()
        if (found) symlinkSync(found, join(tools, tool))
      }
      const executable = (name: string, contents: string) => writeFileSync(join(bin, name), contents, { mode: 0o700 })
      // Only network and privileged package mutation are replaced; parsing and OpenSSL run natively.
      executable("curl", '#!/bin/sh\nwhile [ "$#" -gt 0 ]; do\n case "$1" in --output) output=$2; shift 2;; *) url=$1; shift;; esac\ndone\ncase "$url" in */api/installer?*"&offer=1") cp "$TEST_OFFER" "$output";; *) cp "$TEST_PACKAGE" "$output";; esac\n')
      executable(manager, '#!/bin/sh\nprintf "%s\\n" "$@" > "$TEST_APT_RECEIPT"\n')
      // The installer runs apt-get through env to set a noninteractive frontend.
      executable("env", '#!/bin/sh\nwhile [ "$#" -gt 0 ]; do case "$1" in *=*) shift;; *) break;; esac; done\nexec "$@"\n')
      executable("sudo", '#!/bin/sh\ncase "$1" in -v|-n) exit 0;; esac\nexec "$@"\n')
      const keys = generateKeyPairSync("ed25519")
      const bytes = Buffer.from("verified package fixture")
      const packagePath = join(root, "package.deb"), offerPath = join(root, "offer.json"), receipt = join(root, "receipt")
      writeFileSync(packagePath, scenario === "digest" ? Buffer.from("modified package fixture") : bytes)
      const arch = process.arch === "arm64" ? "arm64" : "x64"
      const release = await Effect.runPromise(signUpdateRelease({ version: scenario === "channel" ? "0.1.6-beta.1" : "0.1.6",
        bytes: bytes.length, sha256: createHash("sha256").update(bytes).digest("hex") },
      { os: "linux", arch: scenario === "architecture" ? (arch === "arm64" ? "x64" : "arm64") : arch, package: format }, keys.privateKey))
      writeFileSync(offerPath, JSON.stringify({ release: scenario === "signature" ? { ...release, signature: "A".repeat(86) + "==" } : release,
        download: `https://github.com/magnitudedev/magnitude/releases/download/test/magnitude.${format}` }))
      const script = await Effect.runPromise(renderUnixInstallationScript({ origin: "https://magnitude.dev", appleTeam: "ABCDEFGHIJ",
        publicKey: keys.publicKey.export({ type: "spki", format: "pem" }).toString() }).pipe(Effect.provide(NodeContext.layer)))
      // setsid: no controlling terminal, as when an agent runs the installer.
      const result = spawnSync(setsid, ["-w", "/bin/sh", "-s"], { input: script, encoding: "utf8", timeout: 15000,
        env: { ...process.env, PATH: `${bin}:${tools}`, TEST_OFFER: offerPath, TEST_PACKAGE: packagePath, TEST_APT_RECEIPT: receipt } })
      if (scenario === "valid") {
        expect(result.status, result.stderr).toBe(0)
        expect(readFileSync(receipt, "utf8")).toMatch(expected)
      } else {
        expect(result.status).not.toBe(0)
        expect(existsSync(receipt)).toBe(false)
      }
    } finally { rmSync(root, { recursive: true, force: true }) }
  })
})

describe.skipIf(process.platform !== "linux")("Linux installation shell architecture admission", () => {
  it("refuses pacman installation on ARM before downloading", () => {
    const root = mkdtempSync(join(tmpdir(), "magnitude-shell-arm-"))
    try {
      const bin = join(root, "bin")
      mkdirSync(bin)
      const executable = (name: string, contents: string) => writeFileSync(join(bin, name), contents, { mode: 0o700 })
      executable("uname", '#!/bin/sh\ncase "$1" in -m) echo aarch64;; *) echo Linux;; esac\n')
      executable("pacman", "#!/bin/sh\nexit 1\n")
      executable("curl", `#!/bin/sh\ntouch ${JSON.stringify(join(root, "downloaded"))}\nexit 1\n`)
      for (const tool of ["python3", "openssl", "mktemp", "rm", "timeout"]) {
        const found = spawnSync("/bin/sh", ["-c", `command -v ${tool}`], { encoding: "utf8" }).stdout.trim()
        if (found) symlinkSync(found, join(bin, tool))
      }
      const script = readFileSync(join(import.meta.dirname, "../../resources/install.sh"), "utf8").replace("@MAGNITUDE_INSTALL_ORIGIN@", "https://magnitude.dev")
      // setsid: no controlling terminal, as when an agent runs the installer.
      const result = spawnSync(setsid, ["-w", "/bin/sh", "-s"], { input: script, encoding: "utf8", timeout: 15000, env: { PATH: bin } })
      expect(result.status).not.toBe(0)
      expect(result.stderr).toContain("x86-64 only")
      expect(existsSync(join(root, "downloaded"))).toBe(false)
    } finally { rmSync(root, { recursive: true, force: true }) }
  })
})

describe.skipIf(process.platform === "win32")("macOS installation shell with the server set up", () => {
  it("refuses before downloading and says how to reinstall", () => {
    const root = mkdtempSync(join(tmpdir(), "magnitude-shell-mac-server-"))
    try {
      const bin = join(root, "bin"), plist = join(root, "dev.magnitude.server.plist")
      mkdirSync(bin)
      writeFileSync(plist, "")
      const executable = (name: string, contents: string) => writeFileSync(join(bin, name), contents, { mode: 0o700 })
      executable("uname", '#!/bin/sh\ncase "$1" in -m) echo arm64;; *) echo Darwin;; esac\n')
      executable("curl", `#!/bin/sh\ntouch ${JSON.stringify(join(root, "downloaded"))}\nexit 1\n`)
      for (const tool of ["mktemp", "rm"]) {
        const found = spawnSync("/bin/sh", ["-c", `command -v ${tool}`], { encoding: "utf8" }).stdout.trim()
        if (found) symlinkSync(found, join(bin, tool))
      }
      const script = readFileSync(join(import.meta.dirname, "../../resources/install.sh"), "utf8").replace("@MAGNITUDE_INSTALL_ORIGIN@", "https://magnitude.dev")
        .replace("mac_server=/Library/LaunchDaemons/dev.magnitude.server.plist", `mac_server=${plist}`)
      const result = spawnSync("/bin/sh", ["-s"], { input: script, encoding: "utf8", timeout: 15000, env: { PATH: bin } })
      expect(result.status).not.toBe(0)
      expect(result.stderr).toContain("magnitude server remove")
      expect(existsSync(join(root, "downloaded"))).toBe(false)
    } finally { rmSync(root, { recursive: true, force: true }) }
  })
})
