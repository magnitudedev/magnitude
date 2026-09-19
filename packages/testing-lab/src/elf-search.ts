import { Effect, Schema } from "effect"
import { posix } from "node:path"
import { AssertionFailure } from "./domain"
import { NativeDependencies } from "./native-dependencies"

export const ElfSearch = Schema.Struct({ direct: Schema.Array(Schema.String), inherited: Schema.Array(Schema.String) })
const fail = (message: string) => new AssertionFailure({ message: `ELF dependency search: ${message}` })

/** Owned search policy: environment paths and system/cache resolution belong to separate stages. */
export const elfSearch = (root: string, file: string, report: Extract<NativeDependencies, { format: "elf" }>, ancestors: readonly string[]) => Effect.gen(function* () {
  const inside = (path: string) => path === root || path.startsWith(root + "/")
  if (!posix.isAbsolute(root) || posix.normalize(root) !== root || !inside(file) || posix.normalize(file) !== file) {
    return yield* fail("expected canonical owned root and file paths")
  }
  if (ancestors.some(path => !inside(path) || posix.normalize(path) !== path)) return yield* fail("inherited search path escapes ownership")
  const expand = (path: string) => Effect.gen(function* () {
    const origin = path === "$ORIGIN" || path === "${ORIGIN}"
      ? "" : path.startsWith("$ORIGIN/") ? path.slice(8) : path.startsWith("${ORIGIN}/") ? path.slice(10) : undefined
    if (origin === undefined || origin.includes("$") || origin.startsWith("/")) {
      return yield* fail(`unqualified or ambient search path in ${file}: ${path}`)
    }
    const expanded = posix.resolve(posix.dirname(file), origin)
    if (!inside(expanded)) return yield* fail(`search path escapes ownership in ${file}: ${path}`)
    return expanded
  })
  // Validate even shadowed RPATH declarations: dormant developer paths must not ship.
  const rpath = yield* Effect.forEach(report.rpaths, expand)
  const runpath = yield* Effect.forEach(report.runpaths, expand)
  const inherited = [...new Set([...(report.runpaths.length === 0 ? rpath : []), ...ancestors])]
  return ElfSearch.make({ direct: [...new Set([...inherited, ...runpath])], inherited })
})
