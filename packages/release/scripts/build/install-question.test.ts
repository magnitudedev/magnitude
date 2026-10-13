import { NodeContext } from "@effect/platform-node"
import { Effect } from "effect"
import { createHash, generateKeyPairSync } from "node:crypto"
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, symlinkSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { spawnSync } from "node:child_process"
import { afterEach, beforeEach, describe, expect, it } from "vitest"
import { signUpdateRelease } from "../../src/hosted-update/release"
import { renderUnixInstallationScript } from "./installation-scripts"

const which = (tool: string) => spawnSync("/bin/sh", ["-c", `command -v ${tool}`], { encoding: "utf8" }).stdout.trim()
const python = which("python3"), setsid = which("setsid")
const tools = ["cat", "cp", "cut", "id", "kill", "mkdir", "mktemp", "openssl", "python3", "rm", "sha256sum", "sh", "sleep", "timeout", "tr", "uname", "wc"]
// Drives the installer in a new session whose controlling terminal is a pty, with the script on
// stdin as `curl | sh` does, and answers the question once it appears.
const driver = `
import json, os, pty, select, sys
script, answer = sys.argv[1:3]
pid, fd = pty.fork()
if pid == 0:
    os.execv("/bin/sh", ["sh", "-c", 'exec sh -s < "$0"', script])
output, answered = b"", False
while True:
    ready, _, _ = select.select([fd], [], [], 20)
    if not ready: break
    try: chunk = os.read(fd, 4096)
    except OSError: break
    if not chunk: break
    output += chunk
    if not answered and b"[y/N]" in output:
        answered = True
        if answer == "<eof>": os.write(fd, b"\\x04")
        elif answer != "<none>": os.write(fd, (answer + "\\n").encode())
_, status = os.waitpid(pid, 0)
print(json.dumps({"output": output.decode(errors="replace"), "status": os.waitstatus_to_exitcode(status)}))
`
const question = "Run Magnitude as a server? It starts on boot, runs without the desktop app,\r\nand you can use it from a browser on another computer. [y/N]"

const arch = process.arch === "x64" || process.arch === "arm64" ? process.arch : undefined

describe.skipIf(process.platform !== "linux" || !arch || !python || !setsid)("Linux installer question", () => {
  let root: string, bin: string, log: string, script: string, environment: Record<string, string>
  beforeEach(async () => {
    root = mkdtempSync(join(tmpdir(), "magnitude-question-"))
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
    stub("systemctl", 'printf "systemctl %s\\n" "$*" >> "$TEST_LOG"\n')
    stub("env", 'while [ "$#" -gt 0 ]; do case "$1" in *=*) shift;; *) break;; esac; done\nexec "$@"\n')
    stub("apt-get", 'printf "install\\n" >> "$TEST_LOG"\n[ -z "${TEST_INSTALL_FAILS:-}" ] || exit 100\n')
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
    writeFileSync(script, await Effect.runPromise(renderUnixInstallationScript({ origin: "https://magnitude.dev", appleTeam: "ABCDEFGHIJ",
      publicKey: keys.publicKey.export({ type: "spki", format: "pem" }).toString() }).pipe(Effect.provide(NodeContext.layer))))
    environment = { PATH: bin, HOME: root, TEST_LOG: log, TEST_OFFER: join(root, "offer.json"), TEST_PACKAGE: join(root, "package.deb") }
  })
  afterEach(() => rmSync(root, { recursive: true, force: true }))

  const events = () => existsSync(log) ? readFileSync(log, "utf8").trimEnd().split("\n") : []
  const answer = (value: string, extra: Record<string, string> = {}) => {
    const result = spawnSync(python, ["-c", driver, script, value], { encoding: "utf8", timeout: 30000, env: { ...environment, ...extra } })
    return JSON.parse(result.stdout) as { output: string; status: number }
  }
  const unattended = (extra: Record<string, string> = {}, args: readonly string[] = []) =>
    spawnSync(setsid, ["-w", "/bin/sh", "-s", "--", ...args], { input: readFileSync(script), encoding: "utf8", timeout: 30000, env: { ...environment, ...extra } })

  it("asks first, gets sudo before any download, then runs server setup after Yes", () => {
    const result = answer("y")
    expect(result.status, result.output).toBe(0)
    expect(result.output).toContain(question)
    const seen = events()
    expect(seen[0]).toBe("sudo -n true")
    expect(seen.findIndex(line => line.startsWith("download"))).toBeGreaterThan(0)
    expect(seen.at(-1)).toBe("magnitude server setup")
    expect(result.output).not.toContain("Skipped")
  })
  it.each([["n", "No"], ["", "Enter"], ["<eof>", "Ctrl-D"]])("skips server setup for %j (%s)", value => {
    const result = answer(value)
    expect(result.status, result.output).toBe(0)
    expect(result.output).toContain("Skipped. To set it up later: magnitude server setup")
    expect(events()).not.toContain("magnitude server setup")
    expect(events()).toContain("install")
  })
  it("treats no answer within 60 seconds as No", () => {
    // The real timeout is exercised on the test machines; here it reports expiry at once.
    rmSync(join(bin, "timeout"), { force: true })
    writeFileSync(join(bin, "timeout"), "#!/bin/sh\nexit 124\n", { mode: 0o700 })
    const result = answer("<none>")
    expect(result.status, result.output).toBe(0)
    expect(result.output).toContain(question)
    expect(result.output).toContain("No answer, skipping. To set it up later: magnitude server setup")
    expect(events()).not.toContain("magnitude server setup")
  })
  it("does not ask without a terminal, uses non-interactive sudo, and prints the Skipped line", () => {
    const result = unattended()
    expect(result.status, result.stderr).toBe(0)
    expect(`${result.stdout}${result.stderr}`).not.toContain("[y/N]")
    expect(result.stdout).toContain("Skipped. To set it up later: magnitude server setup")
    expect(events()[0]).toBe("sudo -n true")
  })
  it("never validates with sudo -v, which asks NOPASSWD accounts for a password under verifypw=all", () => {
    const result = answer("y")
    expect(result.status, result.output).toBe(0)
    expect(result.output).not.toContain("password is required")
    expect(events().filter(line => line.startsWith("sudo")).every(line => !/(^| )-v( |$)/.test(line))).toBe(true)
  })
  it("asks for the password in the terminal when sudo needs one", () => {
    const result = answer("n", { TEST_SUDO_NEEDS_PASSWORD: "1" })
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
  describe("on a set-up server", () => {
    const service = () => events().filter(line => line.includes("systemctl") || line === "install")
    // The marker lives under /etc; point this copy of the script at one the test owns.
    beforeEach(() => {
      writeFileSync(join(root, "server"), "")
      writeFileSync(script, readFileSync(script, "utf8").replace("server_marker=/etc/magnitude/server", `server_marker=${join(root, "server")}`))
    })
    it("upgrades around a stopped service without asking", () => {
      const result = unattended()
      expect(result.status, result.stderr).toBe(0)
      expect(service()).toEqual(["systemctl is-active --quiet magnitude.service", "sudo systemctl stop magnitude.service",
        "systemctl stop magnitude.service", "install", "sudo systemctl start magnitude.service", "systemctl start magnitude.service"])
      expect(result.stdout).toContain("the server restarted")
    })
    it("starts the server again when the package does not install", () => {
      const result = unattended({ TEST_INSTALL_FAILS: "1" })
      expect(result.status).toBe(1)
      expect(result.stderr).toContain("The Magnitude package could not be installed")
      expect(result.stderr).toContain("The Magnitude server was started again.")
      expect(service()).toEqual(["systemctl is-active --quiet magnitude.service", "sudo systemctl stop magnitude.service",
        "systemctl stop magnitude.service", "install", "sudo -n systemctl start magnitude.service", "systemctl start magnitude.service"])
    })
  })
  it("takes no options", () => {
    const result = unattended({}, ["--channel", "beta"])
    expect(result.status).not.toBe(0)
    expect(result.stderr).toContain("takes no options")
    expect(events()).toEqual([])
  })
})
