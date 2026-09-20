import { FileSystem } from "@effect/platform"
import { ReleaseManifestSchema } from "@magnitudedev/release/contracts"
import { Effect, Schema } from "effect"
import { posix } from "node:path"
import { AssertionFailure, Digest, InfrastructureFailure } from "../domain"
import { InstalledApplication } from "../installer"
import { checkedCommand, command } from "../process"
import { admittedRuntimeComposition } from "../runtime-composition"
import { hashFile, PackagePayload } from "./package-payload"

const fail = (message: string) => new AssertionFailure({ message })
const unsupported = (message: string) => new InfrastructureFailure({ operation: "rpm-package-trust", message })
const FileEntry = Schema.Struct({ path: Schema.NonEmptyString.pipe(Schema.filter(path =>
  posix.isAbsolute(path) && posix.normalize(path) === path && path !== "/" && !path.includes("\0"))),
  bytes: Schema.Number.pipe(Schema.int(), Schema.nonNegative()), mode: Schema.Int.pipe(Schema.between(0, 65535)),
  digest: Schema.String, link: Schema.String, flags: Schema.Int.pipe(Schema.nonNegative()) })
// RPM 4.18 supports shescape, but not JSON formatting. Parse its quoted fields as data; never evaluate them.
export const rpmPayloadQuery = "%{FILEDIGESTALGO}\\n[%{FILENAMES:shescape}\\t%{LONGFILESIZES}\\t%{FILEMODES}\\t%{FILEDIGESTS:shescape}\\t%{FILELINKTOS:shescape}\\t%{FILEFLAGS}\\n]"

export const verifyRpmPayload = (app: InstalledApplication) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem, candidate = app.candidate
  if (candidate.target.packageFormat !== "rpm") return yield* unsupported("RPM payload verification requires an RPM candidate")
  if ((yield* hashFile(candidate.path)) !== candidate.artifact.sha256 || Number((yield* fs.stat(candidate.path)).size) !== candidate.artifact.bytes) return yield* fail("Admitted RPM changed before payload verification")
  const query = yield* checkedCommand("rpm", ["-qp", "--qf", rpmPayloadQuery, candidate.path], { env: { LC_ALL: "C" } })
  const newline = query.stdout.indexOf("\n")
  if (query.stdout.slice(0, newline) !== "8") return yield* unsupported("RPM payload verification requires SHA-256 file digests")
  const inventory = query.stdout.slice(newline + 1)
  // Quoted fields may themselves contain tabs, newlines and escaped single quotes.
  const quoted = String.raw`'((?:[^']|'\\'')*)'`
  const row = new RegExp(`${quoted}\\t([0-9]+)\\t([0-9]+)\\t${quoted}\\t${quoted}\\t([0-9]+)\\n`, "y")
  const entries: (typeof FileEntry.Type)[] = []
  let cursor = 0
  while (cursor < inventory.length) {
    row.lastIndex = cursor
    const fields = row.exec(inventory)
    if (!fields) return yield* fail("RPM returned a malformed file inventory")
    const unquote = (index: number) => fields[index]!.replaceAll("'\\''", "'")
    entries.push(yield* Schema.decodeUnknown(FileEntry)({ path: unquote(1), bytes: Number(fields[2]), mode: Number(fields[3]),
      digest: unquote(4), link: unquote(5), flags: Number(fields[6]) }))
    cursor = row.lastIndex
  }
  if (!entries.length || new Set(entries.map(entry => entry.path)).size !== entries.length) return yield* fail("RPM file inventory is empty or contains duplicate paths")
  for (const required of [app.executable, app.cli]) if (!entries.some(entry => entry.path === required)) return yield* fail("RPM inventory does not own the application launcher and bundled CLI")
  const files: (typeof PackagePayload.Type.files[number])[] = []
  for (const entry of entries) {
    if (entry.flags & 64) return yield* unsupported("RPM ghost files require an explicit installed-state policy")
    if ((yield* fs.realPath(posix.dirname(entry.path))) !== posix.dirname(entry.path)) return yield* fail(`Installed RPM path traverses a symlink: ${entry.path}`)
    const link = yield* fs.readLink(entry.path).pipe(Effect.option)
    const kind = entry.mode & 0o170000
    if (kind === 0o120000) {
      if (link._tag !== "Some" || link.value !== entry.link) return yield* fail(`Installed RPM symlink differs: ${entry.path}`)
      continue
    }
    if (link._tag === "Some") return yield* fail(`Installed RPM entry became a symlink: ${entry.path}`)
    const stat = yield* fs.stat(entry.path)
    if (kind === 0o040000) {
      if (stat.type !== "Directory") return yield* fail(`Installed RPM directory changed type: ${entry.path}`)
      continue
    }
    if (kind !== 0o100000 || stat.type !== "File") return yield* fail(`Unexpected installed RPM entry type: ${entry.path}`)
    const expected = yield* Schema.decodeUnknown(Digest)(entry.digest)
    if (Number(stat.size) !== entry.bytes || (yield* hashFile(entry.path)) !== expected) return yield* fail(`Installed RPM payload differs: ${entry.path}`)
    files.push({ path: entry.path, sha256: expected, bytes: entry.bytes })
  }
  if (!files.length) return yield* fail("RPM contains no regular payload files")
  return PackagePayload.make({ version: candidate.version, files })
}).pipe(Effect.mapError(error => error instanceof AssertionFailure || error instanceof InfrastructureFailure ? error : unsupported("Could not compare the installed RPM payload")))

export const inspectRpmIntegrity = (path: string) => Effect.gen(function* () {
  // Native RPM checks all present digests/signatures. This does not pin a publisher.
  const result = yield* command("rpmkeys", ["--checksig", "--verbose", path], { env: { LC_ALL: "C" }, maxOutputBytes: 64 * 1024 })
  if (result.exitCode !== 0 || !/\bdigests?\b[^\n]*\bOK\b/i.test(result.stdout)) return yield* fail("Native RPM integrity or signature verification failed")
  return result.stdout.trim()
})
export const RpmPackageTrust = Schema.Struct({ policy: Schema.Literal("development"), productionTrusted: Schema.Literal(false),
  verification: Schema.NonEmptyString, payload: PackagePayload, runtimeArtifacts: Schema.Array(Schema.String) })
export const inspectRpmPackageTrust = (app: InstalledApplication, release: typeof ReleaseManifestSchema.Type, production: boolean) => Effect.scoped(Effect.gen(function* () {
  if (production) return yield* unsupported("RPM production trust requires an independently configured publisher policy")
  const payload = yield* verifyRpmPayload(app)
  const verification = yield* inspectRpmIntegrity(app.candidate.path)
  const runtime = yield* admittedRuntimeComposition(release, app.candidate.target)
  return RpmPackageTrust.make({ policy: "development", productionTrusted: false, verification, payload, runtimeArtifacts: runtime.artifacts })
}))
