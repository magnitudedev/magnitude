import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises"
import { spawnSync } from "node:child_process"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { afterAll, beforeAll, describe, expect, it } from "vitest"
import { buildLauncher } from "../scripts/build-launcher"

describe.skipIf(process.platform === "win32")("locally packed npm launcher", () => {
  let root: string
  let launcher: string
  let application: string
  let cli: string
  const run = (executable: string, args: string[], env = process.env) => spawnSync(executable, args, { encoding: "utf8", env, timeout: 30000 })
  beforeAll(async () => {
    root = await mkdtemp(join(tmpdir(), "magnitude npm "))
    const pkg = join(root, "package")
    await mkdir(pkg)
    await buildLauncher(join(pkg, "bin"))
    await writeFile(join(pkg, "package.json"), JSON.stringify({ name: "@magnitudedev/cli", version: "9.9.1", bin: { magnitude: "bin/magnitude.js" }, files: ["bin"] }))
    const packed = run("npm", ["pack", pkg, "--pack-destination", root, "--ignore-scripts", "--json"])
    expect(packed.status, packed.stderr).toBe(0)
    const tarball = join(root, JSON.parse(packed.stdout)[0].filename)
    const prefix = join(root, "installed")
    const installed = run("npm", ["install", "--global", "--prefix", prefix, "--offline", "--ignore-scripts", "--no-audit", "--no-fund", tarball])
    expect(installed.status, installed.stderr).toBe(0)
    launcher = join(prefix, "bin/magnitude")
    application = process.platform === "darwin" ? join(root, "Magnitude.app") : join(root, "desktop/magnitude")
    const resources = process.platform === "darwin" ? join(application, "Contents/Resources") : join(root, "desktop/resources")
    const appExecutable = process.platform === "darwin" ? join(application, "Contents/MacOS/Magnitude") : application
    await mkdir(resources, { recursive: true })
    if (process.platform === "darwin") await mkdir(join(application, "Contents/MacOS"))
    await writeFile(appExecutable, "#!/bin/sh\nexit 99\n", { mode: 0o755 })
    cli = join(resources, "magnitude")
  }, 60000)
  afterAll(async () => { if (root) await rm(root, { recursive: true, force: true }) })

  it("uses the desktop CLI version and sees replacement without reinstalling npm", async () => {
    for (const version of ["1.2.3", "1.2.4"]) {
      await writeFile(cli, `#!/bin/sh\nprintf '${version}\\n'\n`, { mode: 0o755 })
      for (const runtime of ["node", "bun"]) {
        const result = run(runtime, [launcher, "--version"], { ...process.env, MAGNITUDE_DESKTOP_PATH: application })
        expect(result.status, result.stderr).toBe(0)
        expect(result.stdout.trim()).toBe(version)
      }
    }
  })
  it("gives installation guidance without downloading a CLI when desktop is absent", () => {
    const result = run(launcher, ["--help"], { ...process.env, MAGNITUDE_DESKTOP_PATH: join(root, "missing"), MAGNITUDE_RELEASE_BASE_URL: "http://127.0.0.1:1" })
    expect(result.status).toBe(1)
    expect(result.stderr).toContain("https://magnitude.dev")
    expect(result.stderr).toContain("desktop is not installed")
    expect(result.stdout).toBe("")
  })
})
