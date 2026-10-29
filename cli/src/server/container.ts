import { FileSystem } from "@effect/platform"
import { Effect, Schema } from "effect"
import { dirname } from "node:path"
import { isDockerInstallation, type ApplicationRuntime } from "@magnitudedev/daemon-management/desktop-native"

/** Whether this is the Docker image's installation, which the image marks in its resources. */
export const isContainer = (runtime: ApplicationRuntime) => runtime._tag === "Installed" && process.platform === "linux"
  ? isDockerInstallation(runtime.resourcesDirectory) : Effect.succeed(false)

export class ContainerDataUnwritable extends Schema.TaggedError<ContainerDataUnwritable>()("ContainerDataUnwritable", { directory: Schema.String }) {
  override get message() {
    return `Magnitude can't write its data directory ${this.directory}, which belongs to another user. `
      + `On the host, run \`chown -R 1000:1000\` on the directory mounted at ${dirname(this.directory)}, or start the container with \`--user <uid>\` of its owner.`
  }
}

/**
 * A volume created by another user, such as a bind mount of a root-owned folder, refuses every write
 * deep inside startup. Checking first replaces that failure with the fix.
 */
export const requireWritableDataDirectory = (directory: string) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  yield* fs.makeDirectory(directory, { recursive: true })
  yield* fs.access(directory, { writable: true })
}).pipe(Effect.catchIf(error => error._tag === "SystemError" && error.reason === "PermissionDenied", () => new ContainerDataUnwritable({ directory })))
