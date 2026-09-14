import { chmod, mkdtemp, readFile, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { Effect, Layer } from "effect"
import { afterEach, describe, expect, it } from "vitest"
import { CliBinaryResolver } from "./cli-binary-resolver"
import { CliProcessSpawner, cliProcessSpawnerLayer } from "./cli-process-spawner"

const roots: string[] = []

afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) =>
    rm(root, { recursive: true, force: true })
  ))
})

const fakeCliSource = `#!/usr/bin/env node
require("node:fs").writeFileSync(process.env.TEST_SPAWN_OUTPUT, JSON.stringify({
  args: process.argv.slice(2),
  custom: process.env.TEST_CUSTOM_VALUE,
  path: process.env.PATH,
  application: process.env.MAGNITUDE_DESKTOP_PATH,
}))
process.exit(Number(process.env.TEST_SPAWN_EXIT ?? "0"))
`

const spawnWith = async (options: {
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

  const spawnerLayer = cliProcessSpawnerLayer({
    args: options.args ?? [],
    environment: {
      PATH: process.env.PATH,
      TEST_SPAWN_OUTPUT: outputPath,
      TEST_SPAWN_EXIT: options.exitWith,
      ...options.environment,
    },
  }).pipe(
    Layer.provide(Layer.succeed(CliBinaryResolver, { resolve: Effect.succeed({ executable: binary, application: "/Applications/Magnitude.app" }) })),
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

describe.skipIf(process.platform === "win32")("CliProcessSpawner", () => {
  it("passes the resolved desktop location and preserves the environment", async () => {
    const { report } = await spawnWith({ environment: {
      MAGNITUDE_DESKTOP_PATH: "/ignored-parent-app", TEST_CUSTOM_VALUE: "custom value",
    } })
    expect(report.application).toBe("/Applications/Magnitude.app")
    expect(report.custom).toBe("custom value")
    expect(report.path).toBe(process.env.PATH)
  })

  it("preserves literal arguments and the child's exit code", async () => {
    const args = ["models", "a b", "$(echo unsafe)", ";", "--json"]
    const { exitCode, report } = await spawnWith({ args, exitWith: "7" })
    expect(report.args).toEqual(args)
    expect(exitCode).toBe(7)
  })
})
