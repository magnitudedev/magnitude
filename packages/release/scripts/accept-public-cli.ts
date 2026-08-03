import {
  mkdir,
  mkdtemp,
  rm,
  symlink,
  writeFile,
} from "node:fs/promises"
import { tmpdir } from "node:os"
import { resolve } from "node:path"
import { run } from "./build/common"

const version = process.env.MAGNITUDE_RELEASE_VERSION?.trim()
const tarball = process.argv[2]
if (!version || !tarball) {
  throw new Error("release version and accepted npm tarball are required")
}

const root = await mkdtemp(resolve(tmpdir(), "magnitude-public-cli-"))
const project = resolve(root, "project")
const home = resolve(root, "home")
try {
  await mkdir(project, { recursive: true, mode: 0o700 })
  await writeFile(resolve(project, "package.json"), "{}\n")
  await run(["npm", "install", "--ignore-scripts", resolve(tarball)], {
    cwd: project,
  })
  const executable = resolve(project, "node_modules/.bin/magnitude")
  const runtimes = [
    { name: "Node.js", command: "node", executable: Bun.which("node") },
    { name: "Bun", command: "bun", executable: process.execPath },
  ] as const
  for (const runtime of runtimes) {
    if (!runtime.executable) {
      throw new Error(`${runtime.name} is required to test the public CLI`)
    }
    const runtimeBin = resolve(root, runtime.command)
    await mkdir(runtimeBin, { mode: 0o700 })
    await symlink(runtime.executable, resolve(runtimeBin, runtime.command))
    const output = await run([executable, "--version"], {
      cwd: project,
      env: {
        ...process.env,
        HOME: home,
        USERPROFILE: home,
        PATH: runtimeBin,
        MAGNITUDE_RELEASE_BASE_URL:
          "https://github.com/magnitudedev/magnitude/releases/download",
      },
    })
    if (output.trim() !== version) {
      throw new Error(
        `public CLI returned ${output.trim()} with ${runtime.name}; expected ${version}`,
      )
    }
  }
} finally {
  await rm(root, { recursive: true, force: true })
}
