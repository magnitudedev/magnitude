import { Effect, Option } from "effect"
import { backendPacks, type HostId } from "../../release/src/targets"
import { Backend, InfrastructureFailure } from "./domain"

/** Apple Silicon startup requires Metal distribution metadata even for CPU execution.
 * Artifact availability is distinct from the backend a generation must attest. */
export const runtimePackBackend = (host: HostId, backend: typeof Backend.Type): Option.Option<"metal" | "cuda"> =>
  backend === "cpu" ? (host === "darwin-arm64" ? Option.some("metal") : Option.none()) : Option.some(backend)

/** The lab exercises the release CUDA 12.9 pack on all CUDA hosts. */
export const selectBuildBackendPacks = (host: HostId, backend: typeof Backend.Type) => Effect.gen(function* () {
  const required = runtimePackBackend(host, backend)
  if (Option.isNone(required)) return []
  const packs = backendPacks.filter(pack => pack.host === host && pack.backend === required.value
    && (pack.backend !== "cuda" || pack.cuda.toolkitVersion === "12.9"))
  if (packs.length !== 1) return yield* new InfrastructureFailure({ operation: "build-backend", message: `No unique required ${required.value} release pack for ${host}` })
  return packs
})
