import { NodeContext } from "@effect/platform-node"
import { Effect } from "effect"
import { createHash, generateKeyPairSync } from "node:crypto"
import { existsSync, lstatSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, readlinkSync, realpathSync, rmSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { spawnSync } from "node:child_process"
import { afterEach, beforeEach, describe, expect, it } from "vitest"
import { signUpdateRelease } from "../../src/hosted-update/release"
import { renderUnixInstallationScript } from "./installation-scripts"

const os = process.platform === "darwin" ? "darwin" : process.platform === "linux" ? "linux" : undefined
const arch = process.arch === "x64" || process.arch === "arm64" ? process.arch : undefined
const finished = (version: string) => `✓ Installed the Magnitude CLI ${version} (CLI only, no desktop app)\n\n`
  + "Next: run `magnitude docs` for how to use `magnitude serve` and how to\n"
  + "keep it running in the background at login on macOS, Linux and Windows.\n"
const quote = (value: string) => `'${value.replaceAll("'", "'\\''")}'`
const payloadFiles = ["magnitude", "magnitude-service", "desktop-host.node", "magnitude-command", "update-configuration.json", "update-trust.json", "LICENSE"]

// The script's own tools run natively; curl serves the offer and archive from files, as a fixed release origin would.
describe.skipIf(!os || !arch)("CLI-only installation script", () => {
  let root: string, home: string, bin: string, log: string, script: string, keys: ReturnType<typeof generateKeyPairSync>
  beforeEach(async () => {
    root = realpathSync(mkdtempSync(join(tmpdir(), "magnitude cli only ' ")))
    home = join(root, "home"), bin = join(root, "bin"), log = join(root, "events")
    mkdirSync(home)
    mkdirSync(bin)
    const stub = (name: string, contents: string) => writeFileSync(join(bin, name), `#!/bin/sh\n${contents}`, { mode: 0o700 })
    stub("curl", 'while [ "$#" -gt 0 ]; do case "$1" in --output) output=$2; shift 2;; *) url=$1; shift;; esac; done\n'
      + 'printf "download %s\\n" "$url" >> "$TEST_LOG"\ncase "$url" in */api/installer?*) cp "$TEST_OFFER" "$output";; *) cp "$TEST_ARCHIVE" "$output";; esac\n')
    stub("codesign", 'printf "codesign %s\\n" "${*##*/}" >> "$TEST_LOG"\nexit "${TEST_CODESIGN_FAILS:-0}"\n')
    keys = generateKeyPairSync("ed25519")
    const rendered = await Effect.runPromise(renderUnixInstallationScript({ origin: "https://magnitude.dev", appleTeam: "ABCDEFGHIJ",
      publicKey: keys.publicKey.export({ type: "spki", format: "pem" }).toString() }).pipe(Effect.provide(NodeContext.layer)))
    // Desktop locations and the code-signature check point into the fixture.
    script = join(root, "install.sh")
    writeFileSync(script, rendered.replace("destination=/Applications/Magnitude.app", `destination=${quote(join(root, "Magnitude.app"))}`)
      .replace("linux_desktop=/usr/lib/magnitude-desktop/resources/magnitude", `linux_desktop=${quote(join(root, "desktop-magnitude"))}`)
      .replaceAll("/usr/bin/codesign", quote(join(bin, "codesign"))))
  })
  afterEach(() => rmSync(root, { recursive: true, force: true }))

  const release = async (version: string, options: { readonly reports?: string; readonly extra?: string; readonly tamper?: boolean; readonly signature?: string } = {}) => {
    const payload = join(root, `payload-${version}`)
    mkdirSync(payload)
    for (const name of payloadFiles) writeFileSync(join(payload, name), `${name}\n`, { mode: 0o755 })
    writeFileSync(join(payload, "magnitude"), [
      "#!/bin/sh",
      `case "$1" in --version) echo ${options.reports ?? version};; *) printf "magnitude %s %s\\n" "${version}" "$*" >> "$TEST_LOG";; esac`,
    ].join("\n") + "\n", { mode: 0o755 })
    const names = [...payloadFiles]
    if (options.extra) { writeFileSync(join(payload, options.extra), ""); names.push(options.extra) }
    const archive = join(root, `magnitude-cli-only-${version}.tar.gz`)
    const packed = spawnSync("tar", ["-czf", archive, "-C", payload, ...names], { env: { ...process.env, COPYFILE_DISABLE: "1" } })
    expect(packed.status, packed.stderr.toString()).toBe(0)
    const bytes = readFileSync(archive)
    const signed = await Effect.runPromise(signUpdateRelease({ version, bytes: bytes.length, sha256: createHash("sha256").update(bytes).digest("hex") },
      { os: os!, arch: arch!, package: "cli-only" }, keys.privateKey))
    if (options.tamper) writeFileSync(archive, Buffer.concat([bytes.subarray(0, -1), Buffer.from([bytes[bytes.length - 1]! ^ 1])]))
    const offer = join(root, `offer-${version}.json`)
    writeFileSync(offer, JSON.stringify({ release: options.signature ? { ...signed, signature: options.signature } : signed,
      download: `https://github.com/magnitudedev/magnitude/releases/download/test/magnitude-cli-only-${os}-${arch}.tar.gz` }))
    return { TEST_OFFER: offer, TEST_ARCHIVE: archive }
  }
  const run = (served: Record<string, string>, args: readonly string[] = [], extra: Record<string, string> = {}) => {
    rmSync(log, { force: true })
    const result = spawnSync("/bin/sh", ["-s", "--", ...args], { input: readFileSync(script), encoding: "utf8", timeout: 60000,
      env: { PATH: `${bin}:/usr/bin:/bin:/usr/sbin:/sbin`, HOME: home, TEST_LOG: log, SHELL: "/bin/bash", ...served, ...extra } })
    const events = existsSync(log) ? readFileSync(log, "utf8").trimEnd().split("\n") : []
    return { ...result, events }
  }
  const cli = () => join(home, ".magnitude", "cli")
  const command = () => join(home, ".magnitude", "bin", "magnitude")

  it("installs into the account's own folders, links the command through current, and registers PATH", async () => {
    const result = run(await release("0.1.6"))
    expect(result.status, result.stderr).toBe(0)
    expect(result.stderr).toContain("Installing the Magnitude CLI only (for servers and headless use). For the desktop app, press Ctrl-C and rerun with --desktop.")
    expect(result.stdout).toBe(`${finished("0.1.6")}Open a new terminal to use the magnitude command.\n`)
    expect(readlinkSync(join(cli(), "current"))).toBe("0.1.6")
    expect(readdirSync(join(cli(), "0.1.6")).sort()).toEqual([...payloadFiles].sort())
    expect(readlinkSync(command())).toBe("../cli/current/magnitude")
    expect(spawnSync(command(), ["--version"], { encoding: "utf8" }).stdout.trim()).toBe("0.1.6")
    expect(result.events.filter(line => line.startsWith("download"))).toEqual([
      `download https://magnitude.dev/api/installer?os=${os}&arch=${arch}&package=cli-only&offer=1`,
      `download https://github.com/magnitudedev/magnitude/releases/download/test/magnitude-cli-only-${os}-${arch}.tar.gz`])
    expect(result.events).toContain("magnitude 0.1.6 _install-cli")
    if (os === "darwin") {
      expect(result.events.filter(line => line.startsWith("codesign")).map(line => line.split(" ").at(-1)))
        .toEqual(["magnitude", "magnitude-service", "desktop-host.node", "magnitude-command"])
    }
    expect(readdirSync(cli()).sort()).toEqual(["0.1.6", "current"])
  })

  it("does not ask for a new terminal when the command folder is already on PATH", async () => {
    const served = await release("0.1.6")
    const result = run(served, [], { PATH: `${join(home, ".magnitude", "bin")}:${bin}:/usr/bin:/bin:/usr/sbin:/sbin` })
    expect(result.status, result.stderr).toBe(0)
    expect(result.stdout).toBe(finished("0.1.6"))
  })

  it("reports an up-to-date install without downloading it again, and upgrades keeping only the previous version", async () => {
    const first = await release("0.1.6")
    expect(run(first).status).toBe(0)
    const again = run(first)
    expect(again.status, again.stderr).toBe(0)
    expect(again.stdout).toContain("The Magnitude CLI 0.1.6 is already up to date (CLI only, no desktop app).")
    expect(again.events.filter(line => line.startsWith("download"))).toHaveLength(1)
    expect(run(await release("0.1.7")).status).toBe(0)
    const third = run(await release("0.1.8"))
    expect(third.status, third.stderr).toBe(0)
    expect(third.stdout).toContain(finished("0.1.8"))
    expect(readdirSync(cli()).sort()).toEqual(["0.1.7", "0.1.8", "current"])
    expect(spawnSync(command(), ["--version"], { encoding: "utf8" }).stdout.trim()).toBe("0.1.8")
    expect(lstatSync(join(cli(), "current")).isSymbolicLink()).toBe(true)
  }, 240_000)

  it.each([
    ["a modified archive", { tamper: true }, "checksum does not match"],
    ["an unexpected archive entry", { extra: "payload.sh" }, "unexpected entry: payload.sh"],
    ["a command reporting another version", { reports: "0.1.5" }, "reports version 0.1.5, not 0.1.6"],
  ] as const)("refuses %s and changes nothing", async (_, options, message) => {
    const result = run(await release("0.1.6", options))
    expect(result.status).not.toBe(0)
    expect(result.stderr).toContain(message)
    expect(existsSync(cli()) ? readdirSync(cli()) : []).toEqual([])
    expect(existsSync(command())).toBe(false)
  })

  it.runIf(os === "linux")("refuses an offer without the publisher's signature before downloading", async () => {
    const result = run(await release("0.1.6", { signature: "A".repeat(86) + "==" }))
    expect(result.status).not.toBe(0)
    expect(result.stderr).toContain("Application release verification failed")
    expect(result.events.filter(line => line.startsWith("download"))).toHaveLength(1)
    expect(existsSync(cli())).toBe(false)
  })

  it.runIf(os === "darwin")("refuses executables without the publisher's Developer ID signature", async () => {
    const result = run(await release("0.1.6"), [], { TEST_CODESIGN_FAILS: "1" })
    expect(result.status).not.toBe(0)
    expect(result.stderr).toContain("does not carry the Magnitude publisher signature")
    expect(readdirSync(cli())).toEqual([])
  })

  it("leaves an installed desktop app alone, since it already includes the command", async () => {
    if (os === "darwin") mkdirSync(join(home, "Applications", "Magnitude.app"), { recursive: true })
    else writeFileSync(join(root, "desktop-magnitude"), "")
    const result = run(await release("0.1.6"))
    expect(result.status, result.stderr).toBe(0)
    expect(result.stdout).toContain("The Magnitude desktop app is installed and already includes the magnitude command. Nothing was changed.")
    expect(result.events).toEqual([])
    expect(existsSync(join(home, ".magnitude"))).toBe(false)
  })

  it("rejects unknown options with both install commands", async () => {
    const result = run(await release("0.1.6"), ["--server"])
    expect(result.status).not.toBe(0)
    expect(result.stderr).toContain("install.sh does not accept: --server.")
    expect(result.stderr).toContain("curl -fsSL https://magnitude.dev/install.sh | sh -s -- --desktop")
    expect(result.events).toEqual([])
  })
})
