import { Effect } from "effect"
import { ApplicationUpdateFailed } from "./application-update"
import { NativeMacUpdate, stageMacUpdateArchive } from "./mac-update-stage"
import { ApplicationUpdateHandoff } from "./update-handoff"
import { hostedUpdateSource, type HostedUpdateSourceOptions } from "./hosted-update-source"

export const macUpdateSource = (options: HostedUpdateSourceOptions) => Effect.gen(function* () {
  const native = yield* NativeMacUpdate
  const handoff = yield* ApplicationUpdateHandoff
  return hostedUpdateSource(options, (archive, candidate) => handoff.record(candidate.manifest.version).pipe(
    Effect.zipRight(stageMacUpdateArchive(archive).pipe(Effect.provideService(NativeMacUpdate, native))),
    Effect.mapError(error => new ApplicationUpdateFailed({ message: error.message })),
  ))
})
