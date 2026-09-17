import { FileSystem } from "@effect/platform"
import { NodeContext } from "@effect/platform-node"
import { Effect } from "effect"
import { join } from "node:path"
import { describe, expect, it } from "vitest"
import { makeMacCliLink } from "./mac-cli-link"

const fixture = <A>(test: (link: ReturnType<typeof makeFixture>, fs: FileSystem.FileSystem) => Effect.Effect<A, unknown, FileSystem.FileSystem>) =>
  Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-cli-link-" })
    const paths = makeFixture(root)
    return yield* test(paths, fs)
  })).pipe(Effect.provide(NodeContext.layer)))

const makeFixture = (root: string) => ({
  root, link: join(root, "bin/magnitude"), target: join(root, "Magnitude's App.app/Contents/Resources/magnitude"),
})

describe("Mac desktop command link", () => {
  it("survives replacement of the bundled CLI and removes only its own link", () => fixture((paths, fs) => Effect.gen(function* () {
    yield* fs.makeDirectory(join(paths.target, ".."), { recursive: true })
    yield* fs.writeFileString(paths.target, "first version")
    const cli = yield* makeMacCliLink({ ...paths })
    expect(yield* cli.read).toBe("Missing")
    yield* cli.install
    yield* cli.install
    expect(yield* cli.read).toBe("Installed")
    yield* fs.remove(paths.target)
    yield* fs.writeFileString(paths.target, "replacement version")
    expect(yield* fs.readFileString(paths.link)).toBe("replacement version")
    yield* cli.remove
    yield* cli.remove
    expect(yield* cli.read).toBe("Missing")
    expect(yield* fs.readFileString(paths.target)).toBe("replacement version")
  })))

  it.each(["file", "foreign-link", "dangling-link"] as const)("replaces a %s command without touching its target", kind => fixture((paths, fs) => Effect.gen(function* () {
    yield* fs.makeDirectory(join(paths.root, "bin"))
    const other = join(paths.root, "other-command")
    if (kind === "file") yield* fs.writeFileString(paths.link, "user command")
    else {
      if (kind === "foreign-link") yield* fs.writeFileString(other, "other installation")
      yield* fs.symlink(other, paths.link)
    }
    const cli = yield* makeMacCliLink({ ...paths })
    expect(yield* cli.read).toBe("Other")
    yield* cli.install
    expect(yield* cli.read).toBe("Installed")
    expect(yield* fs.readLink(paths.link)).toBe(paths.target)
    if (kind === "foreign-link") expect(yield* fs.readFileString(other)).toBe("other installation")
    yield* cli.remove
    expect(yield* cli.read).toBe("Missing")
  })))

  it("preserves a command replaced after installation", () => fixture((paths, fs) => Effect.gen(function* () {
    const cli = yield* makeMacCliLink({ ...paths })
    yield* cli.install
    yield* fs.remove(paths.link)
    yield* fs.writeFileString(paths.link, "user replacement")
    yield* cli.remove
    expect(yield* fs.readFileString(paths.link)).toBe("user replacement")
  })))
  it("replaces PATH commands and cleans up only its remaining links", () => fixture((paths, fs) => Effect.gen(function* () {
    const npmBin = join(paths.root, "npm/bin")
    const missingBin = join(paths.root, "unused/bin")
    const npmCommand = join(npmBin, "magnitude")
    yield* fs.makeDirectory(npmBin, { recursive: true })
    yield* fs.writeFileString(npmCommand, "#!/bin/sh\nexit 0\n")
    yield* fs.chmod(npmCommand, 0o755)
    const cli = yield* makeMacCliLink({ ...paths, path: `${npmBin}:${missingBin}:${npmBin}` })
    yield* cli.install
    expect(yield* fs.readLink(npmCommand)).toBe(paths.target)
    expect(yield* fs.exists(missingBin)).toBe(false)
    yield* cli.install
    yield* fs.remove(npmCommand)
    yield* fs.writeFileString(npmCommand, "later replacement")
    yield* cli.remove
    expect(yield* fs.readFileString(npmCommand)).toBe("later replacement")
    expect(yield* cli.read).toBe("Missing")
  })))

  it("does not remove a directory named magnitude", () => fixture((paths, fs) => Effect.gen(function* () {
    yield* fs.makeDirectory(paths.link, { recursive: true })
    yield* fs.writeFileString(join(paths.link, "keep"), "keep")
    const cli = yield* makeMacCliLink({ ...paths })
    expect((yield* Effect.either(cli.install))._tag).toBe("Left")
    expect(yield* fs.readFileString(join(paths.link, "keep"))).toBe("keep")
  })))
})
