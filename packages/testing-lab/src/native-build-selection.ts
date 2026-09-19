import { Effect } from "effect"
import { backendPacks, type HostId } from "../../release/src/targets"
import { Backend, InfrastructureFailure } from "./domain"

/** The lab exercises the release CUDA 12.9 pack on all CUDA hosts; CPU needs only the base. */
export const selectBuildBackendPacks = (host: HostId, backend: typeof Backend.Type) => Effect.gen(function* () {
  if (backend === "cpu") return []
  const packs = backendPacks.filter(pack => pack.host === host && pack.backend === backend
    && (pack.backend !== "cuda" || pack.cuda.toolkitVersion === "12.9"))
  if (packs.length !== 1) return yield* new InfrastructureFailure({ operation: "build-backend", message: `No unique ${backend} release pack for ${host}` })
  return packs
})
