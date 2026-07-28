import {
  mkdir,
  mkdtemp,
  rm,
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
  const output = await run([
    "node",
    resolve(project, "node_modules/@magnitudedev/cli/bin/magnitude.js"),
    "--version",
  ], {
    cwd: project,
    env: {
      ...process.env,
      HOME: home,
      USERPROFILE: home,
      MAGNITUDE_RELEASE_BASE_URL:
        "https://github.com/magnitudedev/magnitude/releases/download",
    },
  })
  if (output.trim() !== version) {
    throw new Error(`public CLI returned ${output.trim()}; expected ${version}`)
  }
} finally {
  await rm(root, { recursive: true, force: true })
}
