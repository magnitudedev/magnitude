import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Schema } from "effect"
import { join } from "node:path"
import { expect, test } from "vitest"
import { Candidate } from "../src/candidate"
import { targets } from "../src/catalog"
import { InstalledApplication } from "../src/installer"
import { ProcessExecutor } from "../src/process"
import { sha256 } from "../src/snapshot"
import { inspectRpmIntegrity, rpmPayloadQuery, verifyRpmPayload } from "../src/suites/rpm-package-trust"

const target = targets.find(value => value.id === "fedora-44-x64-cpu-intel")!
const quote = (value: string) => `'${value.replaceAll("'", "'\\''")}'`
for (const mode of ["exact", "escaped-name", "changed-file", "changed-link", "file-link", "directory-link", "changed-package",
  "empty", "duplicate", "weak-digest", "ghost", "unsupported-type", "missing-cli"] as const) test(`RPM payload fails closed for ${mode}`, () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-rpm-payload-" }).pipe(Effect.flatMap(fs.realPath))
  const installed = join(root, "installed"), filename = mode === "escaped-name" ? "a file's\twith\nseparators" : "payload"
  yield* fs.makeDirectory(installed)
  const file = join(installed, filename), launcher = join(installed, "launcher"), bytes = "native payload"
  yield* fs.writeFileString(file, mode === "changed-file" ? "native PAYLOAD" : bytes)
  yield* fs.symlink(mode === "changed-link" ? "wrong" : filename, launcher)
  if (mode === "file-link") {
    yield* fs.rename(file, join(root, "outside-file"))
    yield* fs.symlink(join(root, "outside-file"), file)
  }
  if (mode === "directory-link") {
    yield* fs.rename(installed, join(root, "outside-directory"))
    yield* fs.symlink(join(root, "outside-directory"), installed)
  }
  const archive = "admitted RPM", path = join(root, "candidate.rpm")
  yield* fs.writeFileString(path, mode === "changed-package" ? "mutated RPM!" : archive)
  const candidate = yield* Schema.decodeUnknown(Candidate)({ target, version: "1.2.4", path,
    artifact: { id: "desktop-linux-x64-rpm", kind: "desktop", host: "linux-x64-gnu", filename: "Magnitude.rpm", bytes: archive.length, sha256: sha256(archive) } })
  const app = InstalledApplication.make({ candidate, root: installed, executable: launcher, cli: file, packageVersion: "1.2.4-1" })
  const row = (name: string, size: number, permissions: number, digest: string, link: string, flags = 0) =>
    [quote(name), String(size), String(permissions), quote(digest), quote(link), String(flags)].join("\t")
  const entries = [row(installed, 0, 0o040755, "", ""),
    ...(mode === "missing-cli" ? [] : [row(file, bytes.length, mode === "unsupported-type" ? 0o010644 : 0o100644, sha256(bytes), "", mode === "ghost" ? 64 : 0)]),
    row(launcher, filename.length, 0o120777, "", filename)]
  if (mode === "duplicate") entries.push(entries[1]!)
  let queried = false
  const result = yield* verifyRpmPayload(app).pipe(Effect.provideService(ProcessExecutor, { run: spec => Effect.sync(() => {
    queried = true
    expect(spec.executable).toBe("rpm")
    expect(spec.args).toEqual(["-qp", "--qf", rpmPayloadQuery, path])
    return { exitCode: 0, stdout: `${mode === "weak-digest" ? "1" : "8"}\n${mode === "empty" ? "" : entries.join("\n") + "\n"}`, stderr: "" }
  }) }), Effect.either)
  expect(queried).toBe(mode !== "changed-package")
  expect(result._tag).toBe(["exact", "escaped-name"].includes(mode) ? "Right" : "Left")
  if (result._tag === "Right") expect(result.right.files).toEqual([{ path: file, sha256: sha256(bytes), bytes: bytes.length }])
})).pipe(Effect.provide(BunContext.layer))))

test("native RPM verification requires success and digest evidence", async () => {
  for (const [exitCode, stdout, accepted] of [[0, "Header SHA256 digest: OK\nPayload SHA256 digest: OK", true],
    [1, "Header SHA256 digest: OK\nSignature: NOKEY", false], [0, "", false], [0, "unknown output", false]] as const) {
    const result = await Effect.runPromise(inspectRpmIntegrity("/candidate.rpm").pipe(Effect.provideService(ProcessExecutor, {
      run: spec => Effect.sync(() => {
        expect(spec.executable).toBe("rpmkeys")
        expect(spec.args).toEqual(["--checksig", "--verbose", "/candidate.rpm"])
        expect(spec.env.LC_ALL).toBe("C")
        return { exitCode, stdout, stderr: "" }
      }),
    }), Effect.either))
    expect(result._tag).toBe(accepted ? "Right" : "Left")
  }
})
