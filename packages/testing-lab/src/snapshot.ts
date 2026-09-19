import { FileSystem } from "@effect/platform"
import { Effect, Option, Schema } from "effect"
import { createHash } from "node:crypto"
import { dirname, join, posix, resolve, relative, isAbsolute } from "node:path"
import { Digest, InvalidInput } from "./domain"
import { checkedCommand } from "./process"

export const RelativePath = Schema.NonEmptyString.pipe(Schema.filter(path => !path.includes("\\") && !path.includes("\0")
  && !path.startsWith("/") && !/^[A-Za-z]:/.test(path) && path.split("/").every(part => part !== ".." && part !== "." && part !== "" && part.toLowerCase() !== ".git")), Schema.brand("LabRelativePath"))
export const SourceEntry = Schema.Union(
  Schema.Struct({ kind: Schema.Literal("file"), path: RelativePath, sha256: Digest, bytes: Schema.Int.pipe(Schema.nonNegative()), executable: Schema.Boolean }),
  Schema.Struct({ kind: Schema.Literal("symlink"), path: RelativePath, target: Schema.NonEmptyString }),
)
export const SourceManifest = Schema.Struct({ schemaVersion: Schema.Literal(1), kind: Schema.Literal("source"),
  commit: Schema.String.pipe(Schema.pattern(/^[a-f0-9]{40,64}$/)), entries: Schema.Array(SourceEntry),
})
export type SourceManifest = typeof SourceManifest.Type
export const sha256 = (bytes: Uint8Array | string): Digest => Digest.make(createHash("sha256").update(bytes).digest("hex"))
export const manifestJson = (manifest: SourceManifest) => Schema.encodeSync(Schema.parseJson(SourceManifest))(manifest)

const confinedLink = (path: string, target: string) => !isAbsolute(target) && !target.includes("\\") && !/^[A-Za-z]:/.test(target)
  && !posix.normalize(posix.join(posix.dirname(path), target)).split("/").includes("..")
  && !posix.normalize(posix.join(posix.dirname(path), target)).split("/").some(p => p.toLowerCase() === ".git")

export const validateManifest = (manifest: SourceManifest) => Effect.gen(function* () {
  const seen = new Set<string>()
  const folded = new Set<string>()
  const links = new Set(manifest.entries.filter(e => e.kind === "symlink").map(e => e.path as string))
  for (const entry of manifest.entries) {
    yield* Schema.decodeUnknown(RelativePath)(entry.path).pipe(Effect.mapError(() => new InvalidInput({ message: `Unsafe source path: ${entry.path}` })))
    if (seen.has(entry.path) || folded.has(entry.path.toLowerCase())) return yield* new InvalidInput({ message: `Duplicate or case-colliding source path: ${entry.path}` })
    seen.add(entry.path); folded.add(entry.path.toLowerCase())
    if (entry.kind === "symlink" && !confinedLink(entry.path, entry.target)) return yield* new InvalidInput({ message: `Escaping source symlink: ${entry.path}` })
    const parts = entry.path.split("/")
    for (let i = 1; i < parts.length; i++) if (links.has(parts.slice(0, i).join("/"))) return yield* new InvalidInput({ message: `Source writes through a symlink: ${entry.path}` })
  }
})

/** Content objects are named by digest. Workers receive these bytes, never a live checkout. */
export const snapshotSource = (sourceRoot: string, objectRoot: string) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const source = yield* fs.realPath(sourceRoot)
  const destination = resolve(objectRoot)
  const rel = relative(source, destination)
  if (!rel.startsWith(`..${process.platform === "win32" ? "\\" : "/"}`) && rel !== ".." && !isAbsolute(rel)) {
    return yield* new InvalidInput({ message: "Snapshot object storage must be outside the source checkout" })
  }
  yield* fs.makeDirectory(destination, { recursive: true, mode: 0o700 })
  const git = (cwd: string, args: readonly string[]) => checkedCommand("git", args, { cwd: Option.some(cwd), maxOutputBytes: 64 * 1024 * 1024 }).pipe(Effect.map(r => r.stdout))
  const commit = (yield* git(source, ["rev-parse", "HEAD"])).trim()
  const enumerate = (directory: string, prefix = ""): Effect.Effect<readonly (typeof SourceEntry.Type)[], InvalidInput | import("./domain").InfrastructureFailure | import("@effect/platform/Error").PlatformError, FileSystem.FileSystem | import("./process").ProcessExecutor> => Effect.gen(function* () {
    const index = yield* git(directory, ["ls-files", "--stage", "-z"])
    const modules = new Set<string>()
    for (const record of index.split("\0").filter(Boolean)) {
      const match = /^(\d+) [a-f0-9]+ (\d)\t([\s\S]+)$/.exec(record)
      if (!match || match[2] !== "0") return yield* new InvalidInput({ message: "Resolve Git index conflicts before snapshotting" })
      if (match[1] === "160000") modules.add(match[3]!)
    }
    const names = [...new Set((yield* git(directory, ["ls-files", "--cached", "--others", "--exclude-standard", "-z"])).split("\0").filter(Boolean))].sort()
    const entries: (typeof SourceEntry.Type)[] = []
    for (const name of names) {
      const path = yield* Schema.decodeUnknown(RelativePath)(`${prefix}${name}`).pipe(Effect.mapError(() => new InvalidInput({ message: `Unsafe source path: ${name}` })))
      const absolute = join(directory, name)
      if (modules.has(name)) {
        if (!(yield* fs.exists(join(absolute, ".git")))) return yield* new InvalidInput({ message: `Submodule is not initialized: ${path}` })
        entries.push(...yield* enumerate(absolute, `${path}/`)); continue
      }
      // readLink does not follow the link, including dangling links.
      const link = yield* fs.readLink(absolute).pipe(Effect.option)
      if (Option.isSome(link)) { entries.push({ kind: "symlink", path, target: link.value }); continue }
      if (!(yield* fs.exists(absolute))) continue // A tracked local deletion is part of the snapshot.
      const stat = yield* fs.stat(absolute)
      if (stat.type !== "File") return yield* new InvalidInput({ message: `Source contains a nonregular file: ${path}` })
      if (stat.size > 512 * 1024 * 1024) return yield* new InvalidInput({ message: `Source file exceeds 512 MiB: ${path}` })
      const bytes = yield* fs.readFile(absolute)
      const digest = sha256(bytes)
      const objectPath = join(destination, digest)
      if (!(yield* fs.exists(objectPath))) {
        const temporary = `${objectPath}.${crypto.randomUUID()}.tmp`
        yield* fs.writeFile(temporary, bytes, { mode: 0o600, flag: "wx" })
        yield* fs.rename(temporary, objectPath)
      }
      entries.push({ kind: "file", path, sha256: digest, bytes: bytes.byteLength, executable: (stat.mode & 0o111) !== 0 })
    }
    return entries
  })
  const first = yield* enumerate(source)
  const second = yield* enumerate(source)
  if (!Schema.equivalence(Schema.Array(SourceEntry))(first, second) || (yield* git(source, ["rev-parse", "HEAD"])).trim() !== commit) {
    return yield* new InvalidInput({ message: "Source changed during snapshot; retry when writes have settled" })
  }
  const manifest = yield* Schema.decodeUnknown(SourceManifest)({ schemaVersion: 1, kind: "source", commit, entries: first }).pipe(
    Effect.mapError(e => new InvalidInput({ message: String(e) })))
  yield* validateManifest(manifest)
  const json = manifestJson(manifest), digest = sha256(json)
  yield* fs.writeFileString(join(destination, digest), json, { mode: 0o600 })
  return { manifest, digest }
})

/** Requires a newly created destination; no existing symlink or file can redirect writes. */
export const extractSource = (manifest: SourceManifest, objectRoot: string, destination: string) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  yield* validateManifest(manifest)
  if (yield* fs.exists(destination)) return yield* new InvalidInput({ message: "Source extraction requires an absent destination" })
  yield* fs.makeDirectory(destination, { mode: 0o700 })
  for (const entry of manifest.entries) {
    const out = join(destination, entry.path)
    yield* fs.makeDirectory(dirname(out), { recursive: true, mode: 0o700 })
    if (entry.kind === "symlink") yield* fs.symlink(entry.target, out)
    else {
      const bytes = yield* fs.readFile(join(objectRoot, entry.sha256))
      if (bytes.byteLength !== entry.bytes || sha256(bytes) !== entry.sha256) return yield* new InvalidInput({ message: `Corrupt source object: ${entry.path}` })
      yield* fs.writeFile(out, bytes, { flag: "wx", mode: entry.executable ? 0o755 : 0o644 })
    }
  }
})
