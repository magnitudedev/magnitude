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

const makeLayout = async (pnpmLayout: boolean) => {
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
  if (pnpmLayout) await writeFile(join(nodeModules, ".modules.yaml"), "")
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
    const { packageRoot, entrypoint } = await makeLayout(false)
    const installation = await inspect({ entrypoint, environment: {} })
    expect(installation.root).toBe(realpathSync(packageRoot))
    expect(installation.version).toBe("1.2.3")
  })

  it("detects npm, bun, and pnpm ownership", async () => {
    const npm = await makeLayout(false)
    expect((await inspect({
      entrypoint: npm.entrypoint,
      environment: { npm_config_user_agent: "npm/11.0.0 node/v24" },
    })).packageManager).toEqual(Option.some("npm"))

    const bunAgent = await makeLayout(false)
    expect((await inspect({
      entrypoint: bunAgent.entrypoint,
      environment: { npm_config_user_agent: "bun/1.3.0 npm/?" },
    })).packageManager).toEqual(Option.some("bun"))

    const bunExec = await makeLayout(false)
    expect((await inspect({
      entrypoint: bunExec.entrypoint,
      environment: { npm_execpath: "/home/user/.bun/bin/bun" },
    })).packageManager).toEqual(Option.some("bun"))

    const pnpm = await makeLayout(true)
    expect((await inspect({
      entrypoint: pnpm.entrypoint,
      environment: { npm_config_user_agent: "npm/11.0.0 node/v24" },
    })).packageManager).toEqual(Option.some("pnpm"))

    const unknown = await makeLayout(false)
    expect((await inspect({
      entrypoint: unknown.entrypoint,
      environment: {},
    })).packageManager).toEqual(Option.none())
  })

  it("fails with LauncherPackageNotFound for a missing entrypoint", async () => {
    await expect(inspect({
      entrypoint: "/nonexistent/bin/magnitude.js",
      environment: {},
    })).rejects.toThrow("unreadable")
  })
})
