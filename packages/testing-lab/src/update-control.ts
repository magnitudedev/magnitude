import { Effect, Option } from "effect"
import { join } from "node:path"
import { NativeHost } from "../../daemon-management/src/desktop-native"
import { ApplicationControlUnavailable } from "../../daemon-management/src/desktop-native/application-control"

/** Resolve on every observation: a Windows replacement owns a new, private pipe. */
export const updateControlEndpoint = (state: string, windows: boolean) => windows
  ? Effect.gen(function* () {
    const native = yield* NativeHost
    const endpoint = yield* native.inspectEndpoint(state)
    if (Option.isNone(endpoint)) return yield* new ApplicationControlUnavailable({ message: "No automatic update owner has published its control endpoint" })
    return endpoint.value
  })
  : Effect.succeed(join(state, "application.sock"))
