import { createHash } from "node:crypto"
import { chmod, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises"
import {
  spawn,
  spawnSync,
  type ChildProcess,
  type SpawnSyncReturns,
} from "node:child_process"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { gzipSync } from "node:zlib"
import { pack } from "tar-stream"
import { afterAll, beforeAll, describe, expect, it } from "vitest"
import { releaseTag, currentHost } from "@magnitudedev/release"
import { buildLauncher } from "../scripts/build-launcher"

/**
 * End-to-end acceptance without publishing: a local static registry serves a
 * real packument and real packed tarballs, so a real `npm install -g` performs
 * the install and the upgrade exactly as it would against npmjs. The installed
 * launcher then acquires its (fake) native CLI from the same server via
 * MAGNITUDE_RELEASE_BASE_URL.
 */
const V1 = "9.9.1"
const V2 = "9.9.2"
const ARTIFACT_FILENAME = "magnitude-cli.tar.gz"

// Static file server in a node child (segments decoded, so npm's
// /@magnitudedev%2fcli and release-tag paths both map onto directories).
// Port travels through a file: child stdio pipes are not reliably delivered
// when bun test runs from the workspace root, and sockets owned by the bun
// test process are not reliably reachable from other processes.
const STATIC_SERVER_SOURCE = `
const { createServer } = require("node:http")
const { existsSync, readFileSync, writeFileSync } = require("node:fs")
const { join } = require("node:path")
const root = process.env.TEST_STATIC_ROOT
const portPath = process.env.TEST_STATIC_PORT_PATH
const server = createServer((request, response) => {
  const segments = new URL(request.url, "http://registry.test").pathname
    .split("/").filter(Boolean).map(decodeURIComponent)
  const target = join(root, ...segments)
  if (!target.startsWith(root) || !existsSync(target)) {
    response.statusCode = 404
    response.end("not found")
    return
  }
  const name = segments.at(-1) ?? ""
  response.setHeader("content-type", !name.includes(".") || name.endsWith(".json")
    ? "application/json"
    : "application/octet-stream")
  response.end(readFileSync(target))
})
server.listen(0, "127.0.0.1", () => writeFileSync(portPath, String(server.address().port)))
`

// The fake native CLI proves which version ran by writing a file, and
// updates the way the real updater does: by executing the package manager's
// ordinary global install (npm resolves \`latest\` from the registry in its
// environment). TEST_TRIGGER_UPDATE=<version> plays the prompt path's accepted
// offer on a plain launch: only the named version updates, then exits with the
// relaunch code on a protocol match exactly like the real CLI, so the
// relaunched newer version falls through to a normal run.
const fakeCliSource = (version: string) => `#!/usr/bin/env node
const { spawnSync } = require("node:child_process")
if (process.argv[2] === "--version") {
  if (process.env.TEST_REJECT_VERSION_PROBE === "1") process.exit(42)
  console.log(${JSON.stringify(version)})
  process.exit(0)
}
const runNpmUpdate = () =>
  spawnSync("npm", ["install", "-g", "@magnitudedev/cli"], { stdio: "inherit" }).status
if (process.argv[2] === "update") {
  process.exit(runNpmUpdate() ?? 1)
}
if (process.env.TEST_TRIGGER_UPDATE === ${JSON.stringify(version)}) {
  const status = runNpmUpdate()
  if (status === 0 && process.env.MAGNITUDE_LAUNCH_PROTOCOL_VERSION === "1") process.exit(75)
  process.exit(status ?? 1)
}
if (process.env.TEST_TRIGGER_SERVICE_UPDATE === ${JSON.stringify(version)}) {
  const status = runNpmUpdate()
  if (status === 0 && process.env.MAGNITUDE_LAUNCH_PROTOCOL_VERSION === "1") process.exit(76)
  process.exit(status ?? 1)
}
if (process.env.TEST_RELAUNCH_WITHOUT_UPDATE === ${JSON.stringify(version)}) {
  process.exit(process.env.MAGNITUDE_LAUNCH_PROTOCOL_VERSION === "1" ? 75 : 1)
}
require("node:fs").writeFileSync(process.env.TEST_MAGNITUDE_OUTPUT, JSON.stringify({
  version: ${JSON.stringify(version)},
  managedBy: process.env.MAGNITUDE_MANAGED_BY,
  packageRoot: process.env.MAGNITUDE_MANAGED_PACKAGE_ROOT,
  launchProtocolVersion: process.env.MAGNITUDE_LAUNCH_PROTOCOL_VERSION,
  args: process.argv.slice(2),
}))
`

const roots: string[] = []
let serveDirectory: string
let prefix: string
let home: string
let outputPath: string
let registryServer: ChildProcess
let baseUrl: string
let childEnvironment: Record<string, string>

const makeCliArtifact = async (version: string): Promise<Buffer> => {
  const tar = pack()
  tar.entry({ name: "bin/magnitude-cli", mode: 0o755 }, fakeCliSource(version))
  tar.finalize()
  const chunks: Buffer[] = []
  for await (const chunk of tar) chunks.push(chunk as Buffer)
  return gzipSync(Buffer.concat(chunks))
}

const makeNpmTarball = async (
  stagingRoot: string,
  launcherPath: string,
  version: string,
): Promise<Buffer> => {
  const staging = join(stagingRoot, `pack-${version}`)
  const packageDirectory = join(staging, "package")
  await mkdir(join(packageDirectory, "bin"), { recursive: true })
  await writeFile(join(packageDirectory, "package.json"), JSON.stringify({
    name: "@magnitudedev/cli",
    version,
    bin: { magnitude: "bin/magnitude.js" },
    files: ["bin/magnitude.js"],
  }))
  await writeFile(join(packageDirectory, "bin", "magnitude.js"), await readFile(launcherPath))
  await chmod(join(packageDirectory, "bin", "magnitude.js"), 0o755)
  const tarball = join(staging, "cli.tgz")
  const result = spawnSync("tar", ["-czf", tarball, "-C", staging, "package"], { encoding: "utf8" })
  expect(result.status, result.stderr).toBe(0)
  return await readFile(tarball) as Buffer
}

const writeReleaseFiles = async (version: string, artifact: Buffer): Promise<void> => {
  const tagDirectory = join(serveDirectory, ...releaseTag(version).split("/"))
  await mkdir(tagDirectory, { recursive: true })
  await writeFile(join(tagDirectory, ARTIFACT_FILENAME), artifact)
  await writeFile(join(tagDirectory, "magnitude-release.json"), JSON.stringify({
    schemaVersion: 2,
    version,
    acnRevision: 1,
    tag: releaseTag(version),
    sourceCommit: "a".repeat(40),
    artifacts: [{
      id: `cli-${currentHost()}`,
      kind: "cli",
      host: currentHost(),
      filename: ARTIFACT_FILENAME,
      bytes: artifact.byteLength,
      sha256: createHash("sha256").update(artifact).digest("hex"),
    }],
  }))
}

const packumentVersion = (version: string, tarball: Buffer) => ({
  name: "@magnitudedev/cli",
  version,
  bin: { magnitude: "bin/magnitude.js" },
  dist: {
    tarball: `${baseUrl}/tarballs/cli-${version}.tgz`,
    shasum: createHash("sha1").update(tarball).digest("hex"),
    integrity: `sha512-${createHash("sha512").update(tarball).digest("base64")}`,
  },
})

const runInstalledLauncher = (
  cliArguments: ReadonlyArray<string> = [],
  environment: Readonly<Record<string, string>> = {},
): SpawnSyncReturns<string> => spawnSync(
  join(prefix, "bin", "magnitude"),
  [...cliArguments],
  { encoding: "utf8", env: { ...childEnvironment, ...environment } },
)

beforeAll(async () => {
  const root = await mkdtemp(join(tmpdir(), "magnitude-e2e-"))
  roots.push(root)
  serveDirectory = join(root, "serve")
  prefix = join(root, "prefix")
  home = join(root, "home")
  outputPath = join(root, "cli-output.json")
  const cache = join(root, "npm-cache")
  const userconfig = join(root, "npmrc")
  await mkdir(serveDirectory, { recursive: true })
  await mkdir(prefix, { recursive: true })
  await mkdir(home, { recursive: true })
  await mkdir(cache, { recursive: true })
  await writeFile(userconfig, "")

  const launcherPath = await buildLauncher(join(root, "build"))

  const portPath = join(root, "port.txt")
  registryServer = spawn("node", ["-e", STATIC_SERVER_SOURCE], {
    env: {
      ...process.env,
      TEST_STATIC_ROOT: serveDirectory,
      TEST_STATIC_PORT_PATH: portPath,
    },
  })
  let registryServerError = ""
  registryServer.stderr?.on("data", (chunk) => {
    registryServerError += String(chunk)
  })
  const port = await new Promise<string>((resolve, reject) => {
    registryServer.once("error", reject)
    registryServer.once("exit", (code) =>
      reject(new Error(`registry server exited with ${code}: ${registryServerError}`)))
    const poll = setInterval(() => {
      readFile(portPath, "utf8").then((value) => {
        clearInterval(poll)
        resolve(value.trim())
      }, () => {})
    }, 25)
  })
  baseUrl = `http://127.0.0.1:${port}`

  const tarballs: Record<string, Buffer> = {
    [V1]: await makeNpmTarball(root, launcherPath, V1),
    [V2]: await makeNpmTarball(root, launcherPath, V2),
  }
  await mkdir(join(serveDirectory, "tarballs"), { recursive: true })
  for (const version of [V1, V2]) {
    await writeFile(join(serveDirectory, "tarballs", `cli-${version}.tgz`), tarballs[version]!)
    await writeReleaseFiles(version, await makeCliArtifact(version))
  }
  await mkdir(join(serveDirectory, "@magnitudedev"), { recursive: true })
  await writeFile(join(serveDirectory, "@magnitudedev", "cli"), JSON.stringify({
    name: "@magnitudedev/cli",
    "dist-tags": { latest: V2 },
    versions: {
      [V1]: packumentVersion(V1, tarballs[V1]!),
      [V2]: packumentVersion(V2, tarballs[V2]!),
    },
  }))

  // A deliberately minimal environment: a real user shell has none of bun's
  // npm_config_* variables, and npm reads its registry/prefix from these.
  childEnvironment = {
    PATH: process.env.PATH ?? "",
    HOME: home,
    USERPROFILE: home,
    TMPDIR: process.env.TMPDIR ?? "/tmp",
    npm_config_registry: baseUrl,
    npm_config_prefix: prefix,
    npm_config_cache: cache,
    npm_config_userconfig: userconfig,
    npm_config_audit: "false",
    npm_config_fund: "false",
    npm_config_update_notifier: "false",
    MAGNITUDE_RELEASE_BASE_URL: baseUrl,
    TEST_MAGNITUDE_OUTPUT: outputPath,
  }
}, 30000)

afterAll(async () => {
  registryServer?.kill()
  await Promise.all(roots.splice(0).map((root) =>
    rm(root, { recursive: true, force: true })
  ))
})

describe("npm launcher end to end", () => {
  it("installs from a registry and launches the acquired native CLI", async () => {
    const install = spawnSync(
      "npm",
      ["install", "-g", `@magnitudedev/cli@${V1}`],
      { encoding: "utf8", env: childEnvironment },
    )
    expect(install.status, install.stderr).toBe(0)

    const run = runInstalledLauncher()
    expect(run.status, run.stderr).toBe(0)
    const report = JSON.parse(await readFile(outputPath, "utf8"))
    expect(report.version).toBe(V1)
    expect(report.managedBy).toBe("npm")
    expect(report.packageRoot).toContain("node_modules/@magnitudedev/cli")
    expect(report.launchProtocolVersion).toBe("1")
  }, 30000)

  it("launches a published cached CLI without probing its version again", async () => {
    const run = runInstalledLauncher([], { TEST_REJECT_VERSION_PROBE: "1" })
    expect(run.status, run.stderr).toBe(0)
    const report = JSON.parse(await readFile(outputPath, "utf8"))
    expect(report.version).toBe(V1)
  }, 30000)

  it("keeps the published launcher's native CLI responsive to terminal resizes", async () => {
    if (process.platform === "win32") return
    const python = Bun.which("python3")
    if (python === null) return
    const script = join(home, "launcher-resize-check.py")
    const probe = join(home, "native-resize-probe.py")
    await writeFile(probe, `#!/usr/bin/env python3
import os, signal

def report_size(_signal, _frame):
    size = os.get_terminal_size(1)
    print(f"SIZE {size.columns} {size.lines}", flush=True)

signal.signal(signal.SIGWINCH, report_size)
print("READY", flush=True)
while True:
    signal.pause()
`)
    await chmod(probe, 0o700)
    await writeFile(script, `
import errno, fcntl, os, pty, select, signal, struct, sys, termios, time

launcher = sys.argv[1]
pid, fd = pty.fork()
if pid == 0:
    os.execv(launcher, [launcher])

def resize(columns, rows):
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, columns, 0, 0))

buffer = b""
def read_until(needle, timeout):
    global buffer
    deadline = time.monotonic() + timeout
    while needle not in buffer and time.monotonic() < deadline:
        ready, _, _ = select.select([fd], [], [], min(0.1, deadline - time.monotonic()))
        if not ready:
            continue
        try:
            chunk = os.read(fd, 65536)
        except OSError as error:
            if error.errno == errno.EIO:
                break
            raise
        if not chunk:
            break
        buffer += chunk
    if needle not in buffer:
        raise RuntimeError("missing " + repr(needle) + " in " + repr(buffer[-1000:]))

try:
    resize(120, 40)
    read_until(b"READY", 10)
    resize(72, 28)
    read_until(b"SIZE 72 28", 5)
    resize(132, 44)
    read_until(b"SIZE 132 44", 5)
    print("published-launcher-resize-delivered")
finally:
    try:
        os.killpg(pid, signal.SIGTERM)
    except OSError:
        pass
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        pass
    os.close(fd)
`)

    const result = spawnSync(
      python,
      [script, join(prefix, "bin", "magnitude")],
      {
        encoding: "utf8",
        env: { ...childEnvironment, MAGNITUDE_CLI_BINARY: probe },
        timeout: 20_000,
      },
    )
    expect(result.status, `${result.stderr}\n${result.stdout}`).toBe(0)
    expect(result.stdout).toContain("published-launcher-resize-delivered")
  }, 30000)

  it("relaunches into the new version within one invocation after a prompt-path update", async () => {
    const run = runInstalledLauncher([], { TEST_TRIGGER_UPDATE: V1 })
    expect(run.status, run.stderr).toBe(0)
    const report = JSON.parse(await readFile(outputPath, "utf8"))
    expect(report.version).toBe(V2)
    expect(report.managedBy).toBe("npm")
  }, 30000)

  it("starts the service with the new version after an explicit update", async () => {
    const install = spawnSync(
      "npm",
      ["install", "-g", `@magnitudedev/cli@${V1}`],
      { encoding: "utf8", env: childEnvironment },
    )
    expect(install.status, install.stderr).toBe(0)

    const run = runInstalledLauncher([], { TEST_TRIGGER_SERVICE_UPDATE: V1 })
    expect(run.status, run.stderr).toBe(0)
    const report = JSON.parse(await readFile(outputPath, "utf8"))
    expect(report.version).toBe(V2)
    expect(report.args).toEqual(["service", "start"])
  }, 30000)

  it("refuses the relaunch with the floor message when the installation did not change", async () => {
    const run = runInstalledLauncher([], { TEST_RELAUNCH_WITHOUT_UPDATE: V2 })
    expect(run.status, run.stderr).toBe(0)
    expect(run.stderr).toContain("Update installed — run `magnitude` to start the new version.")
  }, 30000)
})
