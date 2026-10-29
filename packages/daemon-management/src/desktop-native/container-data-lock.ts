import { Effect, Option, Schema } from "effect"
import { dirname, join } from "node:path"
import { NativeHost } from "./index"

export class ContainerDataInUse extends Schema.TaggedError<ContainerDataInUse>()("ContainerDataInUse", { directory: Schema.String }) {
  override get message() {
    return `Another Magnitude container is already using the data at ${dirname(this.directory)}. `
      + "Give each container its own volume, or stop the other container first."
  }
}

/**
 * A container's ownership lock and control socket live in its own runtime directory, so they can't
 * see a second container started on the same volume. This lock on the volume can, and is held until
 * the scope closes. A filesystem that can't lock leaves startup to go on, as it does without Docker.
 */
export const holdContainerData = (dataDirectory: string) => Effect.gen(function* () {
  const native = yield* NativeHost
  const lock = yield* native.acquireOwnership(join(dataDirectory, "container.lock")).pipe(
    Effect.catchTag("NativeHostUnavailable", error => Effect.logWarning(`The data volume can't be locked: ${error.message}`).pipe(Effect.as(Option.some({})))))
  if (Option.isNone(lock)) return yield* new ContainerDataInUse({ directory: dataDirectory })
})
