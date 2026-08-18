import * as NodeFileSystem from "@effect/platform-node/NodeFileSystem"
import * as NodePath from "@effect/platform-node/NodePath"
import { realpathSync } from "node:fs"
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { Effect, Layer, Option } from "effect"
import { afterEach, describe, expect, it } from "vitest"
import {
  LauncherInstallationInspector,
  launcherInstallationInspectorLayer,
  type LauncherInstallationInspectorConfig,
} from "./launcher-installation-inspector"

const roots: string[] = []

afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) =>
    rm(root, { recursive: true, force: true })
  ))
})

const makeLayout = async (owner: "none" | "pnpm" | "bun") => {
  const root = await mkdtemp(join(tmpdir(), "magnitude-inspector-"))
  roots.push(root)
  const nodeModules = join(root, "node_modules")
  const packageRoot = join(nodeModules, "@magnitudedev", "cli")
  const binDirectory = join(packageRoot, "bin")
  await mkdir(binDirectory, { recursive: true })
  await writeFile(
    join(packageRoot, "package.json"),
    JSON.stringify({ name: "@magnitudedev/cli", version: "1.2.3" }),
  )
  await writeFile(join(binDirectory, "magnitude.js"), "")
  if (owner === "pnpm") await writeFile(join(nodeModules, ".modules.yaml"), "")
  if (owner === "bun") await writeFile(join(root, "bun.lock"), "")
  return { packageRoot, entrypoint: join(binDirectory, "magnitude.js") }
}

const inspect = (config: LauncherInstallationInspectorConfig) =>
  Effect.runPromise(
    LauncherInstallationInspector.pipe(
      Effect.flatMap((inspector) => inspector.inspect),
      Effect.provide(launcherInstallationInspectorLayer(config).pipe(
        Layer.provide(Layer.mergeAll(NodeFileSystem.layer, NodePath.layer)),
      )),
    ),
  )

describe("LauncherInstallationInspector", () => {
  it("reads the package root and version", async () => {
    const { packageRoot, entrypoint } = await makeLayout("none")
    const installation = await inspect({ entrypoint })
    expect(installation.root).toBe(realpathSync(packageRoot))
    expect(installation.version).toBe("1.2.3")
  })

  it("detects ownership from filesystem markers only", async () => {
    const pnpm = await makeLayout("pnpm")
    expect((await inspect({ entrypoint: pnpm.entrypoint })).packageManager)
      .toEqual(Option.some("pnpm"))

    const bun = await makeLayout("bun")
    expect((await inspect({ entrypoint: bun.entrypoint })).packageManager)
      .toEqual(Option.some("bun"))

    // npm leaves no marker in a global tree; no positive evidence means no
    // claim here, and the spawner's default carries npm.
    const unmarked = await makeLayout("none")
    expect((await inspect({ entrypoint: unmarked.entrypoint })).packageManager)
      .toEqual(Option.none())
  })

  it("fails with LauncherPackageNotFound for a missing entrypoint", async () => {
    await expect(inspect({
      entrypoint: "/nonexistent/bin/magnitude.js",
    })).rejects.toThrow("unreadable")
  })
})
