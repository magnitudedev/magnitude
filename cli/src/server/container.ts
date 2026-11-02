import { FileSystem } from "@effect/platform"
import { Effect, Schema } from "effect"
import { dirname } from "node:path"
import { isDockerInstallation, type ApplicationRuntime } from "@magnitudedev/daemon-management/desktop-native"

/** Whether this is the Docker image's installation, which the image marks in its resources. */
export const isContainer = (runtime: ApplicationRuntime) => runtime._tag === "Installed" && process.platform === "linux"
  ? isDockerInstallation(runtime.resourcesDirectory) : Effect.succeed(false)

export class ContainerDataUnwritable extends Schema.TaggedError<ContainerDataUnwritable>()("ContainerDataUnwritable", { directory: Schema.String }) {
  override get message() {
    return `Magnitude can't write its data directory ${this.directory}. On the host, run \`chown -R 1000:1000\` on the directory mounted at `
      + `${dirname(this.directory)} or start the container with \`--user <uid>\` of its owner; with SELinux, also add \`:Z\` to the volume.`
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

/**
 * A GPU passed with `--device /dev/dri` is usable only by the device's group, which the container's user
 * isn't in unless `--group-add` adds it. Without it the GPU is silently absent, so each unopenable
 * render node gets one line naming the flag.
 */
export const unopenableRenderNodes = (directory = "/dev/dri") => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const nodes = (yield* fs.readDirectory(directory).pipe(Effect.orElseSucceed(() => [] as string[]))).filter(name => name.startsWith("renderD")).sort()
  const refused = yield* Effect.filter(nodes, node => fs.access(`${directory}/${node}`, { readable: true, writable: true }).pipe(
    Effect.as(false), Effect.catchAll(error => Effect.succeed(error._tag === "SystemError" && error.reason === "PermissionDenied"))))
  return refused.map(node => `${directory}/${node}`)
})

export const renderNodeHint = (node: string) =>
  `The GPU device ${node} is in the container, but Magnitude can't open it. Add \`--group-add $(stat -c %g ${node})\` to \`docker run\` to use it.`
