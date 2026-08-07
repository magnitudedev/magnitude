import { readFile, writeFile } from "node:fs/promises"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..", "..")
const revisionPath = resolve(root, "packages/version/acn-revision.json")

export const advanceAcnRevision = async (
  path: string = revisionPath,
): Promise<void> => {
  const record = JSON.parse(await readFile(path, "utf8")) as {
    revision?: unknown
  }
  if (!Number.isSafeInteger(record.revision) || (record.revision as number) <= 0) {
    throw new Error("ACN revision record is malformed")
  }
  const next = (record.revision as number) + 1
  if (!Number.isSafeInteger(next)) throw new Error("ACN revision space is exhausted")
  await writeFile(path, `${JSON.stringify({ revision: next }, null, 2)}\n`)
}

if (import.meta.main) await advanceAcnRevision()
