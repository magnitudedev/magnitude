import * as NodeCommandExecutor from "@effect/platform-node/NodeCommandExecutor"
import * as NodeFileSystem from "@effect/platform-node/NodeFileSystem"
import { chmod, mkdtemp, readFile, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import type { PackageManager } from "@magnitudedev/release"
import { Effect, Layer, Option } from "effect"
import { afterEach, describe, expect, it } from "vitest"
import { CliBinaryResolver, cliBinaryResolverPinnedLayer } from "./cli-binary-resolver"
import { CliProcessSpawner, cliProcessSpawnerLayer } from "./cli-process-spawner"
import { LauncherInstallationInspector } from "./launcher-installation-inspector"

const roots: string[] = []

afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) =>
    rm(root, { recursive: true, force: true })
  ))
})

const fakeCliSource = `#!/usr/bin/env node
require("node:fs").writeFileSync(process.env.TEST_SPAWN_OUTPUT, JSON.stringify({
  args: process.argv.slice(2),
  managedBy: process.env.MAGNITUDE_MANAGED_BY,
  packageRoot: process.env.MAGNITUDE_MANAGED_PACKAGE_ROOT,
  launchProtocolVersion: process.env.MAGNITUDE_LAUNCH_PROTOCOL_VERSION,
  path: process.env.PATH,
}))
process.exit(Number(process.env.TEST_SPAWN_EXIT ?? "0"))
`

const spawnWith = async (options: {
  readonly packageManager: Option.Option<PackageManager>
  readonly args?: ReadonlyArray<string>
  readonly environment?: Readonly<Record<string, string | undefined>>
  readonly exitWith?: string
}) => {
  const root = await mkdtemp(join(tmpdir(), "magnitude-spawner-"))
  roots.push(root)
  const binary = join(root, "fake-cli")
  await writeFile(binary, fakeCliSource)
  await chmod(binary, 0o755)
  const outputPath = join(root, "spawn-output.json")

  const inspectorStub = Layer.succeed(LauncherInstallationInspector, {
    inspect: Effect.succeed({
      root: "/installed/launcher",
      version: "1.0.0",
      packageManager: options.packageManager,
    }),
  })
  const spawnerLayer = cliProcessSpawnerLayer({
    args: options.args ?? [],
    environment: {
      PATH: process.env.PATH,
      TEST_SPAWN_OUTPUT: outputPath,
      TEST_SPAWN_EXIT: options.exitWith,
      ...options.environment,
    },
  }).pipe(
    Layer.provide(inspectorStub),
    Layer.provide(cliBinaryResolverPinnedLayer(binary)),
    Layer.provide(NodeCommandExecutor.layer.pipe(Layer.provide(NodeFileSystem.layer))),
  )

  const exitCode = await Effect.runPromise(
    CliProcessSpawner.pipe(
      Effect.flatMap((spawner) => spawner.spawn),
      Effect.provide(spawnerLayer),
    ),
  )
  const report = JSON.parse(await readFile(outputPath, "utf8"))
  return { exitCode: Number(exitCode), report }
}

describe("CliProcessSpawner", () => {
  it("passes ownership and package root to the native CLI", async () => {
    const { report } = await spawnWith({ packageManager: Option.some("pnpm") })
    expect(report.managedBy).toBe("pnpm")
    expect(report.packageRoot).toBe("/installed/launcher")
    expect(report.path).toBe(process.env.PATH)
  })

  it("declares the launch protocol version it speaks", async () => {
    const { report } = await spawnWith({ packageManager: Option.some("npm") })
    expect(report.launchProtocolVersion).toBe("1")
  })

  it("claims npm when no package manager was detected", async () => {
    const { report } = await spawnWith({ packageManager: Option.none() })
    expect(report.managedBy).toBe("npm")
  })

  it("overrides inherited ownership claims", async () => {
    const { report } = await spawnWith({
      packageManager: Option.some("bun"),
      environment: { MAGNITUDE_MANAGED_BY: "pnpm" },
    })
    expect(report.managedBy).toBe("bun")
  })

  it("passes arguments through and propagates the exit code", async () => {
    const { exitCode, report } = await spawnWith({
      packageManager: Option.some("npm"),
      args: ["--resume", "abc"],
      exitWith: "7",
    })
    expect(report.args).toEqual(["--resume", "abc"])
    expect(exitCode).toBe(7)
  })
})
