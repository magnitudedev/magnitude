import { NodeContext } from "@effect/platform-node"
import { Effect, Option } from "effect"
import { execFileSync } from "node:child_process"
import { chmod, mkdir, mkdtemp, readdir, readlink, rm, symlink, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { afterEach, describe, expect, it } from "vitest"
import { installStandaloneArchive, readStandaloneCurrent, StandaloneCodeVerifier, unsignedStandaloneCode } from "./standalone-installation"

const roots: string[] = []
afterEach(async () => { await Promise.all(roots.splice(0).map(root => rm(root, { recursive: true, force: true }))) })

const workspace = async () => {
  const root = await mkdtemp(join(tmpdir(), "magnitude-standalone-"))
  roots.push(root)
  return root
}

const releaseArchive = async (root: string, version: string, extra: Record<string, string> = {}) => {
  const source = join(root, `source-${version}`)
  await mkdir(source)
  const files: Record<string, string> = {
    magnitude: `#!/bin/sh\necho ${version}\n`, "magnitude-service": "service", "desktop-host.node": "addon",
    "magnitude-command": "command", "update-configuration.json": "{}", "update-trust.json": "{}", LICENSE: "license", ...extra,
  }
  for (const [name, content] of Object.entries(files)) await writeFile(join(source, name), content)
  await chmod(join(source, "magnitude"), 0o755)
  const archive = join(root, `magnitude-cli-only-${version}.tar.gz`)
  execFileSync("tar", ["-czf", archive, "-C", source, ...Object.keys(files)])
  return archive
}

const run = <A, E>(effect: Effect.Effect<A, E, any>) => Effect.runPromise(effect.pipe(
  Effect.provideService(StandaloneCodeVerifier, unsignedStandaloneCode), Effect.provide(NodeContext.layer)) as Effect.Effect<A, E, never>)

describe.skipIf(process.platform === "win32")("standalone installation", () => {
  it("installs beside the running version, switches current and keeps only the two versions", async () => {
    const root = await workspace()
    const installation = join(root, "cli")
    await mkdir(join(installation, "1.0.0"), { recursive: true })
    await mkdir(join(installation, "0.9.0"), { recursive: true })
    await symlink("1.0.0", join(installation, "current"))
    const archive = await releaseArchive(root, "1.1.0")
    const executable = await run(installStandaloneArchive({ root: installation, archive, version: "1.1.0", keep: "1.0.0", platform: "linux", environment: {} }))
    expect(executable).toBe(join(installation, "1.1.0", "magnitude"))
    expect(await readlink(join(installation, "current"))).toBe("1.1.0")
    expect((await readdir(installation)).sort()).toEqual(["1.0.0", "1.1.0", "current"])
    expect(await run(readStandaloneCurrent(installation, "linux"))).toEqual(Option.some("1.1.0"))
  })

  it("refuses a release whose command reports another version and leaves current alone", async () => {
    const root = await workspace()
    const installation = join(root, "cli")
    await mkdir(join(installation, "1.0.0"), { recursive: true })
    await symlink("1.0.0", join(installation, "current"))
    const archive = await releaseArchive(root, "1.0.5")
    const result = await run(Effect.either(installStandaloneArchive({ root: installation, archive, version: "1.1.0", keep: "1.0.0", platform: "linux", environment: {} })))
    expect(result._tag).toBe("Left")
    expect(await readlink(join(installation, "current"))).toBe("1.0.0")
    expect((await readdir(installation)).includes("1.1.0")).toBe(false)
  })

  it("refuses unexpected entries", async () => {
    const root = await workspace()
    const installation = join(root, "cli")
    await mkdir(installation, { recursive: true })
    const archive = await releaseArchive(root, "1.1.0", { "extra.sh": "unexpected" })
    const result = await run(Effect.either(installStandaloneArchive({ root: installation, archive, version: "1.1.0", keep: "1.0.0", platform: "linux", environment: {} })))
    expect(result._tag).toBe("Left")
    expect(await run(readStandaloneCurrent(installation, "linux"))).toEqual(Option.none())
  })
})
