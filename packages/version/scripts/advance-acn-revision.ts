import { readFile, writeFile } from "node:fs/promises"
import { resolve } from "node:path"

const root = resolve(import.meta.dir, "..", "..", "..")
const revisionPath = resolve(root, "packages/version/acn-revision.json")
const packagePath = resolve(root, "packages/cli/package.json")
const bandSize = 1_000_000

const revision = JSON.parse(await readFile(revisionPath, "utf8")) as {
  version?: unknown
  revision?: unknown
}
const packageJson = JSON.parse(await readFile(packagePath, "utf8")) as { version?: unknown }
if (typeof packageJson.version !== "string") throw new Error("CLI package version is missing")
if (typeof revision.version !== "string" ||
  !Number.isSafeInteger(revision.revision) ||
  (revision.revision as number) <= 0 ||
  (revision.revision as number) % bandSize !== 0) {
  throw new Error("ACN revision record is malformed")
}
if (revision.version !== packageJson.version) {
  const next = (revision.revision as number) + bandSize
  if (!Number.isSafeInteger(next)) throw new Error("ACN revision space is exhausted")
  await writeFile(revisionPath, `${JSON.stringify({
    version: packageJson.version,
    revision: next,
  }, null, 2)}\n`)
}
