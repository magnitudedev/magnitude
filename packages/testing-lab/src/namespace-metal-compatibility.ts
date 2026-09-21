import { Effect, Schema } from "effect"
import { isAbsolute, normalize } from "node:path"
import { InfrastructureFailure } from "./domain"

const keys = ["LAB_NAMESPACE_METAL_SHIM", "LAB_NAMESPACE_METAL_FAMILY_MAX", "LAB_NAMESPACE_METAL_MAX_THREADGROUP_MEMORY"] as const
const fail = (message: string) => new InfrastructureFailure({ operation: "namespace-metal-compatibility", message })
const Memory = Schema.NumberFromString.pipe(Schema.int(), Schema.positive())

/** Translate a trusted Namespace preparation receipt into candidate-only loader controls. */
export const namespaceMetalEnvironment = (inherited: Readonly<Record<string, string>>, platform: NodeJS.Platform = process.platform) => Effect.gen(function* () {
  const configured = keys.filter(key => inherited[key] !== undefined)
  if (configured.length === 0) return { ...inherited }
  if (configured.length !== keys.length) return yield* fail("Namespace Metal compatibility configuration is incomplete")
  if (platform !== "darwin") return yield* fail("Namespace Metal compatibility is restricted to macOS workers")
  const shim = inherited.LAB_NAMESPACE_METAL_SHIM!
  if (!isAbsolute(shim) || !normalize(shim).startsWith("/Users/runner/lab-runtime/metal-compatibility/")) {
    return yield* fail("Namespace Metal compatibility shim is outside the prepared runtime")
  }
  if (inherited.LAB_NAMESPACE_METAL_FAMILY_MAX !== "1007") return yield* fail("Namespace Metal compatibility must use the qualified Apple family 7 ceiling")
  const memory = yield* Schema.decodeUnknown(Memory)(inherited.LAB_NAMESPACE_METAL_MAX_THREADGROUP_MEMORY).pipe(
    Effect.mapError(() => fail("Namespace Metal threadgroup-memory receipt is invalid")))
  if (memory !== 32_768) return yield* fail("Namespace Metal compatibility must retain the qualified 32 KiB threadgroup-memory limit")
  if (["DYLD_INSERT_LIBRARIES", "LUME_METAL_APPLE_FAMILY_MAX", "LUME_METAL_MAX_THREADGROUP_MEMORY"].some(key => key in inherited)) {
    return yield* fail("Namespace Metal compatibility conflicts with an existing loader configuration")
  }
  const environment = Object.fromEntries(Object.entries(inherited).filter(([key]) => !keys.includes(key as typeof keys[number])))
  return { ...environment, DYLD_INSERT_LIBRARIES: shim, LUME_METAL_APPLE_FAMILY_MAX: "1007", LUME_METAL_MAX_THREADGROUP_MEMORY: "32768" }
})
