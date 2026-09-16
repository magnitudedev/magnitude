import { FileSystem } from "@effect/platform"
import { Context, Effect, Schema } from "effect"
import { lstat, readlink } from "node:fs/promises"
import { dirname, resolve } from "node:path"

export class CliLinkFailed extends Schema.TaggedError<CliLinkFailed>()("CliLinkFailed", { message: Schema.String }) {}
export type CliLinkState = "Missing" | "Installed" | "Other"
export interface MacCliLink {
  readonly read: Effect.Effect<CliLinkState, CliLinkFailed>
  readonly install: Effect.Effect<void, CliLinkFailed>
  readonly remove: Effect.Effect<void, CliLinkFailed>
}
export const MacCliLink = Context.GenericTag<MacCliLink>("desktop/MacCliLink")
export const makeMacCliLink = (options: {
  readonly link: string
  readonly target: string
  readonly path?: string
  readonly authorize: (link: string, target: string, remove: boolean) => Effect.Effect<void, CliLinkFailed>
}) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const lock = yield* Effect.makeSemaphore(1)
  const candidates = [...new Set([options.link, ...(options.path ?? "").split(":").filter(Boolean)
    .map(directory => resolve(directory, "magnitude")).filter(path => path !== options.target)])]
  // lstat also finds dangling links; unlinking a link never touches its former target.
  const readAt = (link: string) => Effect.tryPromise({ try: async (): Promise<CliLinkState> => {
    const info = await lstat(link).catch((error: NodeJS.ErrnoException) => {
      if (error.code === "ENOENT") return undefined
      throw error
    })
    if (!info) return "Missing"
    return info.isSymbolicLink() && resolve(dirname(link), await readlink(link)) === options.target ? "Installed" : "Other"
  }, catch: error => new CliLinkFailed({ message: `Could not inspect ${link}: ${String(error)}` }) })
  const read = readAt(options.link)
  const install = lock.withPermits(1)(Effect.forEach(candidates, link => Effect.gen(function* () {
    const state = yield* readAt(link)
    if (state === "Installed" || (state === "Missing" && link !== options.link)) return
    const create = Effect.gen(function* () {
      yield* fs.makeDirectory(dirname(link), { recursive: true })
      // Directories are never commands and are never recursively removed.
      if (state !== "Missing") yield* fs.remove(link)
      yield* fs.symlink(options.target, link)
    })
    yield* create.pipe(Effect.catchAll(error => error._tag === "SystemError" && error.reason === "PermissionDenied"
      ? options.authorize(link, options.target, false)
      : Effect.fail(new CliLinkFailed({ message: `Could not install ${link}: ${error.message}` }))))
    if ((yield* readAt(link)) !== "Installed") return yield* new CliLinkFailed({ message: `The command-line link was not installed at ${link}.` })
  }), { discard: true }))
  const remove = lock.withPermits(1)(Effect.forEach(candidates, link => Effect.gen(function* () {
    if ((yield* readAt(link)) !== "Installed") return
    yield* fs.remove(link).pipe(Effect.catchAll(error => error._tag === "SystemError" && error.reason === "PermissionDenied"
      ? options.authorize(link, options.target, true)
      : Effect.fail(new CliLinkFailed({ message: `Could not remove ${link}: ${error.message}` }))))
    if ((yield* readAt(link)) === "Installed") return yield* new CliLinkFailed({ message: `The command-line link could not be removed at ${link}.` })
  }), { discard: true }))
  return MacCliLink.of({ read, install, remove })
})
