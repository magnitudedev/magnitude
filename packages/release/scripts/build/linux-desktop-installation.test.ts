import { NodeContext } from "@effect/platform-node"
import { Effect } from "effect"
import { createHash, generateKeyPairSync } from "node:crypto"
import { existsSync, mkdirSync, mkdtempSync, readFileSync, readlinkSync, rmSync, symlinkSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { spawnSync } from "node:child_process"
import { afterEach, beforeEach, describe, expect, it } from "vitest"
import { signUpdateRelease } from "../../src/hosted-update/release"
import { renderUnixInstallationScript } from "./installation-scripts"

const which = (tool: string) => spawnSync("/bin/sh", ["-c", `command -v ${tool}`], { encoding: "utf8" }).stdout.trim()
const python = which("python3"), setsid = which("setsid")
const tools = ["awk", "date", "cat", "chmod", "cp", "cut", "getconf", "id", "kill", "ln", "mkdir", "mktemp", "mv", "openssl", "python3", "readlink", "rm", "sha256sum", "sh", "sleep", "tr", "uname", "wc"]
// Runs the installer in a new session whose controlling terminal is a pty, with the script on stdin as `curl | sh` does.
const driver = `
import json, os, pty, select, sys
script = sys.argv[1]
pid, fd = pty.fork()
if pid == 0:
    os.execv("/bin/sh", ["sh", "-c", 'exec sh -s -- --desktop < "$0"', script])
output = b""
while True:
    ready, _, _ = select.select([fd], [], [], 20)
    if not ready: break
    try: chunk = os.read(fd, 4096)
    except OSError: break
    if not chunk: break
    output += chunk
_, status = os.waitpid(pid, 0)
print(json.dumps({"output": output.decode(errors="replace"), "status": os.waitstatus_to_exitcode(status)}))
`

const arch = process.arch === "x64" || process.arch === "arm64" ? process.arch : undefined

describe.skipIf(process.platform !== "linux" || !arch || !python || !setsid)("Linux desktop installation", () => {
  let root: string, bin: string, log: string, script: string, environment: Record<string, string>
  beforeEach(async () => {
    root = mkdtempSync(join(tmpdir(), "magnitude-desktop-"))
    bin = join(root, "bin")
    log = join(root, "events")
    mkdirSync(bin)
    for (const tool of tools) { const found = which(tool); if (found) symlinkSync(found, join(bin, tool)) }
    // Stubs replace links to real tools; unlinking first never writes through a link to the system copy.
    const stub = (name: string, contents: string) => { rmSync(join(bin, name), { force: true }); writeFileSync(join(bin, name), `#!/bin/sh\n${contents}`, { mode: 0o700 }) }
    stub("curl", 'while [ "$#" -gt 0 ]; do case "$1" in --output) output=$2; shift 2;; *) url=$1; shift;; esac; done\nprintf "download %s\\n" "$url" >> "$TEST_LOG"\ncase "$url" in */api/installer?*) cp "$TEST_OFFER" "$output";; *) cp "$TEST_PACKAGE" "$output";; esac\n')
    // Behaves like sudo on standard cloud images (verifypw=all): an account with NOPASSWD also matches a
    // rule that needs a password, so `-v` asks for one even though running a command never does.
    stub("sudo", [
      'printf "sudo %s\\n" "$*" >> "$TEST_LOG"',
      'case "$*" in',
      '  -v|"-n -v") echo "sudo: a password is required" >&2; exit 1;;',
      '  "-n true") [ -z "${TEST_SUDO_REFUSES:-}${TEST_SUDO_NEEDS_PASSWORD:-}" ] || exit 1; exit 0;;',
      '  true) [ -z "${TEST_SUDO_REFUSES:-}" ] || exit 1; exit 0;;',
      'esac',
      '[ "$1" != -n ] || shift',
      'exec "$@"',
    ].join("\n") + "\n")
    stub("systemctl", 'printf "systemctl %s\\n" "$*" >> "$TEST_LOG"\n[ "$1 ${TEST_SERVICE_INACTIVE:-}" != "is-active 1" ] || exit 3\n')
    stub("dpkg-query", '[ -n "${TEST_INSTALLED:-}" ] || exit 1\nprintf "%s" "$(cat "$TEST_INSTALLED")"\n')
    stub("env", 'while [ "$#" -gt 0 ]; do case "$1" in *=*) shift;; *) break;; esac; done\nexec "$@"\n')
    stub("apt-get", 'printf "%s\\n" "$1" >> "$TEST_LOG"\n[ "$1" != install ] || [ -z "${TEST_INSTALL_FAILS:-}" ] || exit 100\n[ "$1" != install ] || [ -z "${TEST_INSTALLED:-}" ] || printf "0.1.6" > "$TEST_INSTALLED"\n')
    stub("magnitude", 'printf "magnitude %s\\n" "$*" >> "$TEST_LOG"\n')
    // The installer behaves as an ordinary account, whatever account runs the tests.
    stub("id", 'case "$1" in -u) echo 1000;; *) exec /usr/bin/id "$@";; esac\n')
    const keys = generateKeyPairSync("ed25519")
    const bytes = Buffer.from("verified package fixture")
    writeFileSync(join(root, "package.deb"), bytes)
    const release = await Effect.runPromise(signUpdateRelease({ version: "0.1.6", bytes: bytes.length, sha256: createHash("sha256").update(bytes).digest("hex") },
      { os: "linux", arch: arch ?? "x64", package: "deb" }, keys.privateKey))
    writeFileSync(join(root, "offer.json"), JSON.stringify({ release, download: "https://github.com/magnitudedev/magnitude/releases/download/test/magnitude.deb" }))
    script = join(root, "install.sh")
    writeFileSync(script, (await Effect.runPromise(renderUnixInstallationScript({ origin: "https://magnitude.dev", appleTeam: "ABCDEFGHIJ",
      publicKey: keys.publicKey.export({ type: "spki", format: "pem" }).toString() }).pipe(Effect.provide(NodeContext.layer))))
      .replace("server_marker=/etc/magnitude/server", `server_marker=${join(root, "server")}`))
    environment = { PATH: bin, HOME: root, TEST_LOG: log, TEST_OFFER: join(root, "offer.json"), TEST_PACKAGE: join(root, "package.deb") }
  })
  afterEach(() => rmSync(root, { recursive: true, force: true }))

  const events = () => existsSync(log) ? readFileSync(log, "utf8").trimEnd().split("\n") : []
  const interactive = (extra: Record<string, string> = {}) => {
    const result = spawnSync(python, ["-c", driver, script], { encoding: "utf8", timeout: 30000, env: { ...environment, ...extra } })
    return JSON.parse(result.stdout) as { output: string; status: number }
  }
  const unattended = (extra: Record<string, string> = {}) =>
    spawnSync(setsid, ["-w", "/bin/sh", "-s", "--", "--desktop"], { input: readFileSync(script), encoding: "utf8", timeout: 30000, env: { ...environment, ...extra } })

  it("gets sudo before any download, installs without asking anything, and opens the desktop", () => {
    const result = interactive({ DISPLAY: ":0" })
    expect(result.status, result.output).toBe(0)
    expect(result.output).not.toContain("[y/N]")
    const seen = events()
    expect(seen[0]).toBe("sudo -n true")
    expect(seen.findIndex(line => line.startsWith("download"))).toBeGreaterThan(0)
    expect(seen).toContain("install")
    expect(seen.at(-1)).toBe("magnitude app open")
    expect(seen.some(line => line.includes("server"))).toBe(false)
  })
  it("does not open the desktop over SSH, even with X forwarding", () => {
    const result = interactive({ DISPLAY: "localhost:10.0", SSH_CONNECTION: "fixture" })
    expect(result.status, result.output).toBe(0)
    expect(events()).not.toContain("magnitude app open")
  })
  it("names the version it installed or upgraded from, without the package revision", () => {
    const installed = join(root, "installed")
    expect(unattended({ TEST_INSTALLED: installed }).stdout).toContain("Magnitude 0.1.6 was installed.")
    writeFileSync(installed, "0.1.5-58")
    expect(unattended({ TEST_INSTALLED: installed }).stdout).toContain("Magnitude was upgraded from 0.1.5 to 0.1.6.")
  })
  it("uses non-interactive sudo without a terminal and leaves a headless desktop closed", () => {
    const result = unattended()
    expect(result.status, result.stderr).toBe(0)
    expect(result.stdout).toContain("Magnitude was installed.")
    expect(events()).not.toContain("magnitude app open")
    expect(events()[0]).toBe("sudo -n true")
  })
  it("never validates with sudo -v, which asks NOPASSWD accounts for a password under verifypw=all", () => {
    const result = interactive()
    expect(result.status, result.output).toBe(0)
    expect(result.output).not.toContain("password is required")
    expect(events().filter(line => line.startsWith("sudo")).every(line => !/(^| )-v( |$)/.test(line))).toBe(true)
  })
  it("asks for the password in the terminal when sudo needs one", () => {
    const result = interactive({ TEST_SUDO_NEEDS_PASSWORD: "1" })
    expect(result.status, result.output).toBe(0)
    expect(events().slice(0, 2)).toEqual(["sudo -n true", "sudo true"])
    expect(events()).toContain("install")
  })
  it("fails at once without a terminal or passwordless sudo, before downloading", () => {
    const result = unattended({ TEST_SUDO_REFUSES: "1" })
    expect(result.status).not.toBe(0)
    expect(result.stderr).toContain("no terminal to ask for a password")
    expect(events().some(line => line.startsWith("download"))).toBe(false)
  })
  it("replaces a CLI-only install, pointing its command at the package's", () => {
    mkdirSync(join(root, ".magnitude/cli/0.1.5"), { recursive: true })
    symlinkSync("0.1.5", join(root, ".magnitude/cli/current"))
    mkdirSync(join(root, ".magnitude/bin"))
    symlinkSync("../cli/current/magnitude", join(root, ".magnitude/bin/magnitude"))
    const result = unattended()
    expect(result.status, result.stderr).toBe(0)
    expect(existsSync(join(root, ".magnitude/cli"))).toBe(false)
    expect(readlinkSync(join(root, ".magnitude/bin/magnitude"))).toBe("/usr/bin/magnitude")
  })
  describe("on a server set up by an earlier release", () => {
    const service = () => events().filter(line => line.includes("systemctl") || line === "update" || line === "install")
    // The marker lives under /etc; point this copy of the script at one the test owns.
    beforeEach(() => {
      writeFileSync(join(root, "server"), "")
      environment.DISPLAY = ":0"
      writeFileSync(join(root, "installed"), "0.1.5")
      environment.TEST_INSTALLED = join(root, "installed")
    })
    it("upgrades around a stopped service and starts it again, which converts it", () => {
      const result = unattended()
      expect(result.status, result.stderr).toBe(0)
      expect(service()).toEqual(["update", "systemctl is-active --quiet magnitude.service", "sudo systemctl stop magnitude.service",
        "systemctl stop magnitude.service", "install", "sudo systemctl start magnitude.service", "systemctl start magnitude.service"])
      expect(result.stdout).toContain("Magnitude was upgraded from 0.1.5 to 0.1.6.")
      expect(result.stdout).toContain("The Magnitude server is restarting.")
      expect(events()).not.toContain("magnitude app open")
    })
    it("leaves a server the administrator stopped stopped", () => {
      const result = unattended({ TEST_SERVICE_INACTIVE: "1" })
      expect(result.status, result.stderr).toBe(0)
      expect(service()).toEqual(["update", "systemctl is-active --quiet magnitude.service", "install"])
      expect(result.stdout).toContain("left stopped")
    })
    it("starts the server again when the package does not install", () => {
      const result = unattended({ TEST_INSTALL_FAILS: "1" })
      expect(result.status).toBe(1)
      expect(result.stderr).toContain("The Magnitude package could not be installed")
      expect(result.stderr).toContain("The Magnitude server was started again.")
      expect(service()).toEqual(["update", "systemctl is-active --quiet magnitude.service", "sudo systemctl stop magnitude.service",
        "systemctl stop magnitude.service", "install", "sudo -n systemctl start magnitude.service", "systemctl start magnitude.service"])
    })
  })
})
