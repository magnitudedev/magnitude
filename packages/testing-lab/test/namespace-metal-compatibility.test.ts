import { Effect } from "effect"
import { expect, test } from "vitest"
import { namespaceMetalEnvironment } from "../src/namespace-metal-compatibility"

const configured = {
  PATH: "/usr/bin",
  LAB_NAMESPACE_METAL_SHIM: "/Users/runner/lab-runtime/metal-compatibility/LumeMetalCapabilities-arm64.dylib",
  LAB_NAMESPACE_METAL_FAMILY_MAX: "1007",
  LAB_NAMESPACE_METAL_MAX_THREADGROUP_MEMORY: "32768",
}

test("qualified Namespace Metal configuration stays inert in the candidate environment", async () => {
  expect(await Effect.runPromise(namespaceMetalEnvironment(configured, "darwin"))).toEqual(configured)
})

test("ordinary environments remain unchanged", async () => {
  expect(await Effect.runPromise(namespaceMetalEnvironment({ PATH: "/usr/bin" }, "darwin"))).toEqual({ PATH: "/usr/bin" })
})

test("partial, broadened, conflicting and non-macOS profiles are rejected", async () => {
  for (const [environment, platform] of [
    [{ LAB_NAMESPACE_METAL_SHIM: configured.LAB_NAMESPACE_METAL_SHIM }, "darwin"],
    [{ ...configured, LAB_NAMESPACE_METAL_FAMILY_MAX: "1009" }, "darwin"],
    [{ ...configured, LAB_NAMESPACE_METAL_MAX_THREADGROUP_MEMORY: "65536" }, "darwin"],
    [{ ...configured, DYLD_INSERT_LIBRARIES: "/tmp/other.dylib" }, "darwin"],
    [{ ...configured, LUME_METAL_PROCESS_NAME: "other" }, "darwin"],
    [configured, "linux"],
  ] as const) expect((await Effect.runPromise(namespaceMetalEnvironment(environment, platform).pipe(Effect.either)))._tag).toBe("Left")
})
