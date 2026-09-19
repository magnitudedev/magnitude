import { describe, expect, test } from "vitest"
import { BunContext } from "@effect/platform-bun"
import { FileSystem } from "@effect/platform"
import { Effect, Option, Schema } from "effect"
import { join } from "node:path"
import { ProcessExecutorLive, checkedCommand } from "../src/process"
import { extractSource, RelativePath, sha256, snapshotSource, SourceManifest, validateManifest } from "../src/snapshot"

const run = <A, E>(effect: Effect.Effect<A, E, FileSystem.FileSystem | import("../src/process").ProcessExecutor | import("effect/Scope").Scope>) =>
  Effect.runPromise(Effect.scoped(effect).pipe(Effect.provide([BunContext.layer, ProcessExecutorLive])))

describe("immutable source inputs", () => {
  test("captures dirty files, additions, deletion, executable mode and dirty submodules without ignored data", () => run(Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-snapshot-test-" })
    const repo = join(root, "repo"), sub = join(root, "sub"), objects = join(root, "objects")
    const git = (cwd: string, ...args: string[]) => checkedCommand("git", args, { cwd: Option.some(cwd), env: { GIT_CONFIG_NOSYSTEM: "1" } })
    for (const cwd of [repo, sub]) {
      yield* fs.makeDirectory(cwd)
      yield* git(cwd, "init", "-q")
      yield* git(cwd, "config", "user.name", "Lab Fixture")
      yield* git(cwd, "config", "user.email", "lab@example.invalid")
      yield* fs.writeFileString(join(cwd, "tracked"), "initial")
      yield* git(cwd, "add", ".")
      yield* git(cwd, "commit", "-qm", "Initial fixture")
    }
    yield* git(repo, "-c", "protocol.file.allow=always", "submodule", "add", sub, "nested")
    yield* fs.writeFileString(join(repo, "deleted"), "delete me")
    yield* fs.writeFileString(join(repo, ".gitignore"), "secret\n")
    yield* git(repo, "add", ".")
    yield* git(repo, "commit", "-qm", "Add submodule")
    yield* fs.remove(join(repo, "deleted"))
    yield* fs.writeFileString(join(repo, "tracked"), "local unpublished change")
    yield* fs.writeFileString(join(repo, "new-script"), "#!/bin/sh\ntrue\n")
    yield* fs.chmod(join(repo, "new-script"), 0o755)
    yield* fs.writeFileString(join(repo, "secret"), "must not leave source")
    yield* fs.writeFileString(join(repo, "nested/tracked"), "dirty submodule")
    yield* fs.symlink("tracked", join(repo, "relative-link"))
    const result = yield* snapshotSource(repo, objects)
    const again = yield* snapshotSource(repo, objects)
    expect(result.digest).toBe(again.digest)
    expect(result.manifest.entries.map(e => e.path)).not.toEqual(expect.arrayContaining(["secret", "deleted", ".git"]))
    yield* extractSource(result.manifest, objects, join(root, "consumer"))
    expect(yield* fs.readFileString(join(root, "consumer/tracked"))).toBe("local unpublished change")
    expect(yield* fs.readFileString(join(root, "consumer/nested/tracked"))).toBe("dirty submodule")
    expect(yield* fs.readLink(join(root, "consumer/relative-link"))).toBe("tracked")
    expect((yield* fs.stat(join(root, "consumer/new-script"))).mode & 0o111).not.toBe(0)
  })))

  test.each(["../outside", "/absolute", "C:/windows", "foo/../../outside", "foo\\bar", ".git/config", "foo//bar"])("rejects unsafe path %s", path => {
    expect(Schema.is(RelativePath)(path)).toBe(false)
  })
  test("rejects symlink escapes and writes through symlink parents", async () => {
    for (const entries of [
      [{ kind: "symlink", path: "link", target: "../outside" }],
      [{ kind: "symlink", path: "link", target: "inside" }, { kind: "file", path: "link/file", sha256: sha256("x"), bytes: 1, executable: false }],
    ]) {
      const manifest = Schema.decodeUnknownSync(SourceManifest)({ schemaVersion: 1, kind: "source", commit: "a".repeat(40), entries })
      expect(await Effect.runPromise(validateManifest(manifest).pipe(Effect.either))).toMatchObject({ _tag: "Left" })
    }
  })
  test("rejects corrupted content and an existing extraction destination", () => run(Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-corrupt-test-" })
    const manifest = yield* Schema.decodeUnknown(SourceManifest)({ schemaVersion: 1, kind: "source", commit: "a".repeat(40), entries: [
      { kind: "file", path: "file", sha256: sha256("good"), bytes: 4, executable: false },
    ] })
    yield* fs.writeFileString(join(root, sha256("good")), "evil")
    expect(yield* extractSource(manifest, root, root).pipe(Effect.either)).toMatchObject({ _tag: "Left" })
    expect(yield* extractSource(manifest, root, join(root, "out")).pipe(Effect.either)).toMatchObject({ _tag: "Left" })
  })))
})
