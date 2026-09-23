import { describe, expect, it } from "vitest"
import { mkdtemp, mkdir, writeFile, symlink, rm } from "node:fs/promises"
import { join } from "node:path"
import { tmpdir } from "node:os"
import { NodeContext } from "@effect/platform-node"
import { Effect } from "effect"
import { applicationNativeHostPath, applicationServiceCommand, resolveApplicationProfile, resolveInstalledApplicationRuntime, type ApplicationRuntime } from "./application-bootstrap"

const installed: ApplicationRuntime = { _tag: "Installed", resourcesDirectory: "/Applications/Magnitude.app/Contents/Resources" }
const development: ApplicationRuntime = { _tag: "Development", repository: "/work/Magnitude Source" }

describe("application bootstrap selection", () => {
  it.skipIf(process.platform === "win32")("resolves a symlink chain to its own installed resources", async () => {
    const root = await mkdtemp(join(tmpdir(), "magnitude-runtime-"))
    try {
      const resourcesDirectory = join(root, "Magnitude Test.app", "Contents", "Resources")
      await mkdir(resourcesDirectory, { recursive: true })
      await writeFile(join(resourcesDirectory, "magnitude"), "fixture")
      await symlink(join(resourcesDirectory, "magnitude"), join(root, "first"))
      await symlink(join(root, "first"), join(root, "magnitude"))
      const runtime = await Effect.runPromise(resolveInstalledApplicationRuntime(join(root, "magnitude"), "darwin").pipe(Effect.provide(NodeContext.layer)))
      expect(runtime._tag).toBe("Installed")
      // macOS canonicalizes /var to /private/var when resolving the fixture.
      expect(runtime.resourcesDirectory.endsWith("Magnitude Test.app/Contents/Resources")).toBe(true)
      await writeFile(join(root, "standalone"), "fixture")
      const invalid = await Effect.runPromise(resolveInstalledApplicationRuntime(join(root, "standalone"), "darwin").pipe(Effect.either, Effect.provide(NodeContext.layer)))
      expect(invalid._tag === "Left" && invalid.left._tag).toBe("ApplicationRuntimeUnavailable")
    } finally { await rm(root, { recursive: true, force: true }) }
  })

  it("preserves production, development and acceptance profile isolation", () => {
    const select = (runtime: ApplicationRuntime, acceptance = false, environment: Record<string, string> = {}) =>
      resolveApplicationProfile({ runtime, acceptance, environment, home: "/home/user", platform: "linux" })
    expect(select(installed)).toEqual({ dataDirectory: "/home/user/.magnitude", isolated: false, port: 10100, endpoint: "http://127.0.0.1:10100" })
    expect(select(development)).toMatchObject({ dataDirectory: "/home/user/.magnitude-desktop-dev", isolated: true, port: 11101 })
    expect(select(installed, true)).toMatchObject({ dataDirectory: "/home/user/.magnitude-update-acceptance", isolated: true, port: 11143 })
    expect(select(installed, false, { MAGNITUDE_DEV_DATA_DIR: "/isolated", MAGNITUDE_DEV_PORT: "12345" }))
      .toEqual({ dataDirectory: "/isolated", isolated: true, port: 12345, endpoint: "http://127.0.0.1:12345" })
    expect(select(installed, false, { MAGNITUDE_DEV_PORT: "12345" }).port).toBe(10100)
  })

  it.each([
    ["darwin", "/Applications/Magnitude.app/Contents/Resources", "/Applications/Magnitude.app/Contents/Resources/magnitude-service"],
    ["linux", "/usr/lib/magnitude-desktop/resources", "/usr/lib/magnitude-desktop/resources/magnitude-service"],
    ["win32", "C:\\Users\\Test User\\Magnitude\\resources", "C:\\Users\\Test User\\Magnitude\\resources\\magnitude-service.exe"],
  ])("uses matched installed resources on %s", (platform, resourcesDirectory, executable) => {
    const runtime: ApplicationRuntime = { _tag: "Installed", resourcesDirectory }
    const environment = Object.freeze({ KEEP: "unchanged", MAGNITUDE_ICN_PATH: "/explicit/engine" })
    const profile = resolveApplicationProfile({ runtime, platform, home: platform === "win32" ? "C:\\Users\\Test User" : "/home/user", acceptance: false, environment })
    const command = applicationServiceCommand({ output: "DiagnosticTail", runtime, profile, platform, architecture: "x64", environment })
    expect(command.executable).toBe(executable)
    expect(command.arguments).toEqual(["serve", "--data-dir", profile.dataDirectory, "--port", "10100"])
    expect(command.environment).toEqual({ ...environment, MAGNITUDE_NATIVE_HOST: applicationNativeHostPath(runtime, platform, "x64") })
    expect(environment).not.toHaveProperty("MAGNITUDE_NATIVE_HOST")
  })

  it("runs development source with the chosen runtime and preserves explicit engine selection", () => {
    const profile = resolveApplicationProfile({ runtime: development, home: "/home/user", platform: "darwin", acceptance: false, environment: {} })
    const options = { output: "DiagnosticTail" as const, runtime: development, profile, platform: "darwin", architecture: "arm64" }
    const command = applicationServiceCommand({ ...options, environment: { MAGNITUDE_BUN_PATH: "/tools/bun" } })
    expect(command.executable).toBe("/tools/bun")
    expect(command.arguments[0]).toBe("/work/Magnitude Source/packages/acn/src/binary.ts")
    expect(command.environment.MAGNITUDE_ICN_PATH).toBe("/work/Magnitude Source/inference/target/development/installation.json")
    expect(command.environment.MAGNITUDE_NATIVE_HOST).toBe("/work/Magnitude Source/packages/daemon-management/dist/native/darwin-arm64/desktop-host.node")
    expect(applicationServiceCommand({ ...options, environment: { MAGNITUDE_ICN_PATH: "/external/engine" } }).environment.MAGNITUDE_ICN_PATH).toBe("/external/engine")
  })
})
