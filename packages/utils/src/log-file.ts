import * as FileSystem from "@effect/platform/FileSystem"
import * as Path from "@effect/platform/Path"
import { Chunk, Effect, Exit, Queue, Ref, Scope } from "effect"

export interface LogFile {
  readonly append: (text: string | Uint8Array) => Effect.Effect<void>
}

const encoder = new TextEncoder()

/**
 * A diagnostic log of at most `maxBytes`, rotated once to `<file>.1`. Appending never blocks or fails
 * the caller: output queues (dropping when the disk falls behind) and a scoped fiber writes it, and a
 * file that cannot be opened or written is skipped. Closing the scope writes what is still queued.
 */
export const openLogFile = (file: string, maxBytes: number) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const path = yield* Path.Path
  const queue = yield* Queue.dropping<Uint8Array>(4096)
  const current = yield* Ref.make<{ readonly handle: FileSystem.File; readonly scope: Scope.CloseableScope; readonly size: number } | null>(null)
  const close = Ref.getAndSet(current, null).pipe(Effect.flatMap(open => open ? Scope.close(open.scope, Exit.void) : Effect.void))
  const open = (rotate: boolean) => Effect.gen(function* () {
    yield* fs.makeDirectory(path.dirname(file), { recursive: true, mode: 0o700 })
    const size = yield* fs.stat(file).pipe(Effect.map(info => Number(info.size)), Effect.orElseSucceed(() => 0))
    const rotated = size > 0 && (rotate || size >= maxBytes)
    if (rotated) {
      // Windows cannot rename over an existing file.
      yield* fs.remove(`${file}.1`).pipe(Effect.ignore)
      yield* fs.rename(file, `${file}.1`)
    }
    const scope = yield* Scope.make()
    const handle = yield* fs.open(file, { flag: "a", mode: 0o600 }).pipe(Scope.extend(scope), Effect.onError(() => Scope.close(scope, Exit.void)))
    yield* Ref.set(current, { handle, scope, size: rotated ? 0 : size })
  })
  const write = (chunks: ReadonlyArray<Uint8Array>) => Effect.gen(function* () {
    for (const chunk of chunks) {
      const state = yield* Ref.get(current)
      if (state && state.size > 0 && state.size + chunk.length > maxBytes) {
        yield* close
        yield* open(true)
      }
      const target = yield* Ref.get(current)
      if (!target) return
      yield* target.handle.writeAll(chunk)
      yield* Ref.set(current, { ...target, size: target.size + chunk.length })
    }
  }).pipe(Effect.catchAll(() => close))
  const opened = yield* open(false).pipe(Effect.as(true), Effect.orElseSucceed(() => false))
  if (!opened) return { append: () => Effect.void } satisfies LogFile
  yield* Effect.addFinalizer(() => Queue.takeAll(queue).pipe(Effect.flatMap(rest => write(Chunk.toReadonlyArray(rest))), Effect.zipRight(close)))
  yield* Effect.uninterruptibleMask(restore => restore(Queue.take(queue)).pipe(
    Effect.flatMap(first => Queue.takeAll(queue).pipe(Effect.flatMap(rest => write([first, ...Chunk.toReadonlyArray(rest)])))),
  )).pipe(Effect.forever, Effect.forkScoped)
  return {
    append: text => Queue.offer(queue, typeof text === "string" ? encoder.encode(text) : text).pipe(Effect.asVoid),
  } satisfies LogFile
})
