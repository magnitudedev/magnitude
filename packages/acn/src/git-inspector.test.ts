import { chmod, mkdir, mkdtemp, writeFile } from "node:fs/promises"
import { join } from "node:path"
import { tmpdir } from "node:os"
import { afterEach, beforeAll, describe, expect, it } from "vitest"
import { Effect, Layer } from "effect"
import { DirectoryPathSchema } from "@magnitudedev/acn-protocol"
import { GitInspector, GitInspectorLive } from "./git-inspector"
import { testPlatformLayer } from "./session-test-support"

const cwd = DirectoryPathSchema.make("/repos/alpha")

const FAKE_GIT = `#!/bin/sh
sub="$3"
case "$MAGNITUDE_TEST_GIT_MODE" in
  not_repo)
    echo "fatal: not a git repository (or any of the parent directories): .git" 1>&2
    exit 128 ;;
  broken)
    echo "fatal: unexpected breakage" 1>&2
    exit 1 ;;
  branch)
    if [ "$sub" = "rev-parse" ] && [ "$4" = "--show-toplevel" ]; then echo "/repos/alpha"; exit 0; fi
    if [ "$sub" = "symbolic-ref" ]; then echo "main"; exit 0; fi
    exit 1 ;;
  detached)
    if [ "$sub" = "rev-parse" ] && [ "$4" = "--show-toplevel" ]; then echo "/repos/alpha"; exit 0; fi
    if [ "$sub" = "symbolic-ref" ]; then exit 1; fi
    if [ "$sub" = "rev-parse" ] && [ "$4" = "--short" ]; then echo "abc1234"; exit 0; fi
    exit 1 ;;
  log)
    if [ "$sub" = "log" ]; then printf "a.ts\\n\\nb.ts\\na.ts\\nc.ts\\n"; exit 0; fi
    exit 1 ;;
esac
exit 1
`

let fakeBin = ""
let emptyBin = ""
const originalPath = process.env.PATH
const originalMode = process.env.MAGNITUDE_TEST_GIT_MODE

beforeAll(async () => {
  const root = await mkdtemp(join(tmpdir(), "magnitude-git-inspector-"))
  fakeBin = join(root, "bin")
  emptyBin = join(root, "empty")
  await mkdir(fakeBin)
  await mkdir(emptyBin)
  await writeFile(join(fakeBin, "git"), FAKE_GIT)
  await chmod(join(fakeBin, "git"), 0o755)
})

afterEach(() => {
  if (originalPath !== undefined) process.env.PATH = originalPath
  if (originalMode === undefined) delete process.env.MAGNITUDE_TEST_GIT_MODE
  else process.env.MAGNITUDE_TEST_GIT_MODE = originalMode
})

const layer = GitInspectorLive.pipe(Layer.provide(testPlatformLayer))

const inspect = () => Effect.runPromise(
  GitInspector.pipe(
    Effect.flatMap((git) => git.inspect(cwd)),
    Effect.provide(layer),
  ),
)

const recentFiles = (limit: number) => Effect.runPromise(
  GitInspector.pipe(
    Effect.flatMap((git) => git.recentFiles(cwd, limit)),
    Effect.provide(layer),
  ),
)

describe("GitInspector", () => {
  it("reports a repository with its branch", async () => {
    process.env.PATH = fakeBin
    process.env.MAGNITUDE_TEST_GIT_MODE = "branch"
    expect(await inspect()).toEqual({
      _tag: "git_repository",
      rootDirectory: "/repos/alpha",
      head: { _tag: "branch", name: "main" },
    })
  })

  it("reports a detached head by revision", async () => {
    process.env.PATH = fakeBin
    process.env.MAGNITUDE_TEST_GIT_MODE = "detached"
    expect(await inspect()).toEqual({
      _tag: "git_repository",
      rootDirectory: "/repos/alpha",
      head: { _tag: "detached", revision: "abc1234" },
    })
  })

  it("classifies not-a-repository from the stable diagnostic", async () => {
    process.env.PATH = fakeBin
    process.env.MAGNITUDE_TEST_GIT_MODE = "not_repo"
    expect(await inspect()).toEqual({ _tag: "not_git_repository" })
  })

  it("classifies other command failures as inspection failures", async () => {
    process.env.PATH = fakeBin
    process.env.MAGNITUDE_TEST_GIT_MODE = "broken"
    expect(await inspect()).toEqual({ _tag: "git_inspection_failed" })
  })

  it("reports git as unavailable when the executable cannot be started", async () => {
    process.env.PATH = emptyBin
    expect(await inspect()).toEqual({ _tag: "git_unavailable" })
  })

  it("returns deduplicated recent files bounded to the limit", async () => {
    process.env.PATH = fakeBin
    process.env.MAGNITUDE_TEST_GIT_MODE = "log"
    expect(await recentFiles(2)).toEqual({
      _tag: "recent_git_files",
      files: ["a.ts", "b.ts"],
    })
  })

  it("never swallows recent-file outcomes into an empty list", async () => {
    process.env.PATH = fakeBin
    process.env.MAGNITUDE_TEST_GIT_MODE = "not_repo"
    expect(await recentFiles(5)).toEqual({ _tag: "not_git_repository" })
  })
})
