import { readFile } from "node:fs/promises"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { advanceAcnRevision } from "./advance-acn-revision"

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..", "..")
const packagePath = resolve(root, "packages/cli/package.json")

const readCliVersion = async (): Promise<string> => {
  const packageJson = JSON.parse(await readFile(packagePath, "utf8")) as {
    version?: unknown
  }
  if (typeof packageJson.version !== "string") {
    throw new Error("CLI package version is missing")
  }
  return packageJson.version
}

const before = await readCliVersion()
const changesets = Bun.spawn(["bunx", "changeset", "version"], {
  cwd: root,
  stdin: "inherit",
  stdout: "inherit",
  stderr: "inherit",
})
const exitCode = await changesets.exited
if (exitCode !== 0) process.exit(exitCode)

if (await readCliVersion() !== before) await advanceAcnRevision()
