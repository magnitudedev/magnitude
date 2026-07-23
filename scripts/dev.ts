import { access } from "node:fs/promises"
import { constants } from "node:fs"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { buildLocalIcn } from "../inference/scripts/build-local"

const PROJECT_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..")

const run = async (
  command: string[],
  env: Record<string, string | undefined> = process.env,
): Promise<number> => {
  const child = Bun.spawn(command, {
    cwd: PROJECT_ROOT,
    env,
    stdin: "inherit",
    stdout: "inherit",
    stderr: "inherit",
  })
  const interrupt = () => child.kill("SIGINT")
  const terminate = () => child.kill("SIGTERM")
  process.once("SIGINT", interrupt)
  process.once("SIGTERM", terminate)
  try {
    return await child.exited
  } finally {
    process.off("SIGINT", interrupt)
    process.off("SIGTERM", terminate)
  }
}

const versionExit = await run([
  "bun",
  "run",
  "packages/version/scripts/generate-version.ts",
  "--dev",
])
if (versionExit !== 0) process.exit(versionExit)

const explicit = process.env.MAGNITUDE_ICN_PATH?.trim()
let binaryPath: string
if (explicit) {
  binaryPath = resolve(explicit)
  await access(binaryPath, constants.X_OK)
  console.log(`[dev] Using explicit ICN binary: ${binaryPath}`)
} else {
  const built = await buildLocalIcn()
  binaryPath = built.binaryPath
  console.log(`[dev] Using ${built.backend} ICN binary: ${binaryPath}`)
}

const clientExit = await run(
  ["bun", "run", "cli/src/index.tsx", "--debug", ...process.argv.slice(2)],
  {
    ...process.env,
    MAGNITUDE_ICN_PATH: binaryPath,
  },
)
process.exit(clientExit)
