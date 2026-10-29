import { Effect, Option, Schema } from "effect"
import { dirname, join } from "node:path"
import { NativeHost } from "./index"

export class ContainerDataInUse extends Schema.TaggedError<ContainerDataInUse>()("ContainerDataInUse", { directory: Schema.String }) {
  override get message() {
    return `Another Magnitude container is already using the data at ${dirname(this.directory)}. `
      + "Give each container its own volume, or stop the other container first."
  }
}

export class ContainerDataLockUnsafe extends Schema.TaggedError<ContainerDataLockUnsafe>()("ContainerDataLockUnsafe", { lock: Schema.String }) {
  override get message() {
    return `Magnitude can't use ${this.lock}, which another user created. Run the container as the user that owns the volume, `
      + "or on the host, run `chown -R 1000:1000` on the directory mounted at /data."
  }
}

/** The native lock's answer when the filesystem itself can't lock, as some network filesystems can't. */
const lockingUnsupported = "Cannot acquire ownership lock"

/**
 * A container's ownership lock and control socket live in its own runtime directory, so they can't
 * see a second container started on the same volume. This lock on the volume can, and is held until
 * the scope closes. A filesystem that can't lock leaves startup to go on, as it does without Docker;
 * a lock file this user can't safely use is refused.
 */
export const holdContainerData = (dataDirectory: string) => Effect.gen(function* () {
  const native = yield* NativeHost
  const path = join(dataDirectory, "container.lock")
  const lock = yield* native.acquireOwnership(path).pipe(Effect.catchTag("NativeHostUnavailable", error => error.message === lockingUnsupported
    ? Effect.logWarning(`The data volume can't be locked, so nothing stops a second container from using it: ${error.message}`).pipe(Effect.as(Option.some({})))
    : new ContainerDataLockUnsafe({ lock: path })))
  if (Option.isNone(lock)) return yield* new ContainerDataInUse({ directory: dataDirectory })
})
