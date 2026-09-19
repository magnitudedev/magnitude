import { Effect, Schema } from "effect"
import { AssertionFailure } from "./domain"
import { checkedCommand } from "./process"

const Need = Schema.Struct({ library: Schema.NonEmptyString, versions: Schema.Array(Schema.NonEmptyString) })
export const ElfVersions = Schema.Struct({ definitions: Schema.Array(Schema.String), needs: Schema.Array(Need) })
const fail = (message: string) => new AssertionFailure({ message: `ELF symbol versions: ${message}` })

/** Version names are opaque ABI identities; numerical comparison is insufficient. */
export const decodeElfVersions = (output: string) => Effect.gen(function* () {
  if (output.trim() === "No version information found in this file.") return ElfVersions.make({ definitions: [], needs: [] })
  const definitions: string[] = [], needs: { library: string; versions: string[] }[] = []
  let section: "symbols" | "definitions" | "needs" | undefined
  let expectedDefinitions = 0, expectedNeeds = 0, neededCount = 0
  const finishNeed = () => !needs.length || needs[needs.length - 1]!.versions.length === neededCount
  let found = false
  for (const line of output.split("\n")) {
    const heading = /^Version (symbols|definition|needs) section '[^']+' contains (\d+) entr(?:y|ies):$/.exec(line.trim())
    if (heading) {
      if (section === "needs" && !finishNeed()) return yield* fail("truncated version requirement group")
      found = true
      section = heading[1] === "definition" ? "definitions" : heading[1] === "needs" ? "needs" : "symbols"
      if (section === "definitions") expectedDefinitions += Number(heading[2])
      if (section === "needs") expectedNeeds += Number(heading[2])
      continue
    }
    if (section === "definitions" && /\bRev:/.test(line)) {
      const name = /\bName:\s+(\S+)\s*$/.exec(line)?.[1]
      if (!name) return yield* fail("malformed version definition")
      definitions.push(name)
    }
    if (section === "needs" && /\bFile:/.test(line)) {
      if (!finishNeed()) return yield* fail("truncated version requirement group")
      const file = /\bFile:\s+(\S+)\s+Cnt:\s+(\d+)\s*$/.exec(line)
      if (!file || Number(file[2]) === 0) return yield* fail("malformed version requirement owner")
      needs.push({ library: file[1]!, versions: [] }); neededCount = Number(file[2])
    } else if (section === "needs" && /\bName:/.test(line)) {
      const name = /\bName:\s+(\S+)\s+Flags:/.exec(line)?.[1]
      if (!name || !needs.length) return yield* fail("version requirement has no library")
      needs[needs.length - 1]!.versions.push(name)
    }
  }
  if (!found || definitions.length !== expectedDefinitions || needs.length !== expectedNeeds || !finishNeed()) return yield* fail("missing or truncated version report")
  return ElfVersions.make({ definitions, needs })
})

export const inspectElfVersions = (path: string) => checkedCommand("readelf", ["--wide", "--version-info", path], {
  env: { PATH: "/usr/sbin:/usr/bin:/sbin:/bin", LC_ALL: "C" }, inheritEnv: false, timeoutMs: 30_000, maxOutputBytes: 8 * 1024 * 1024,
}).pipe(Effect.flatMap(output => decodeElfVersions(output.stdout)))

export const attestElfVersions = (library: string, required: readonly string[], available: typeof ElfVersions.Type) => Effect.gen(function* () {
  const missing = required.filter(version => !available.definitions.includes(version))
  if (missing.length) return yield* fail(`${library} does not define required versions: ${missing.join(", ")}`)
})
