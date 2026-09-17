import { mkdtemp, mkdir, realpath, rm, symlink, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { Effect } from "effect"
import { expect, it } from "vitest"
import { resolveMacApplicationPath } from "./mac-application-path"

it("resolves a symlinked CLI to its own application rather than another installed app", async () => {
  const root = await mkdtemp(join(tmpdir(), "magnitude-bundled-cli-"))
  try {
    const bundle = join(root, "Applications/Magnitude.app")
    const resources = join(bundle, "Contents/Resources")
    await mkdir(resources, { recursive: true })
    const cli = join(resources, "magnitude")
    await writeFile(cli, "fixture")
    const link = join(root, "magnitude")
    await symlink(cli, link)
    expect(await Effect.runPromise(resolveMacApplicationPath(link, root))).toBe(await realpath(bundle))
    expect(await Effect.runPromise(resolveMacApplicationPath(cli, root))).toBe(await realpath(bundle))
    expect(await Effect.runPromise(resolveMacApplicationPath(link, root, "/explicit/Magnitude.app"))).toBe("/explicit/Magnitude.app")
  } finally { await rm(root, { recursive: true, force: true }) }
})
