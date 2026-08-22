/**
 * Host files as observed by the file panel: resolve and read queries kept
 * fresh by `WatchFile`. Freshness is a dependency of observation — the watch
 * for a file is open exactly while one of its queries is observed — so no
 * component mounts a watch and no selection is tracked here.
 */
import { Atom, Registry } from "@effect-atom/atom-react"
import type { RpcClientError } from "@effect/rpc/RpcClientError"
import { Context, Effect, Fiber, Layer, Runtime, Stream } from "effect"
import { Key, QueryClient, Subscription, type Query } from "@magnitudedev/effect-query"
import {
  Files as FilesBoundary,
  type ReadFileFormat,
  type ReadFileResult,
  type ResolvePathResult,
} from "@magnitudedev/sdk"
import { ClientEffectQuery } from "../state/client-effect-query"

export interface FileInput {
  readonly cwd: string
  readonly path: string
}

const memoized = <Input, A extends object>(key: (input: Input) => string, make: (input: Input) => A) => {
  const entries = new Map<string, A>()
  return (input: Input): A => {
    const id = key(input)
    const existing = entries.get(id)
    if (existing !== undefined) return existing
    const created = make(input)
    entries.set(id, created)
    return created
  }
}

const makeFiles = Effect.gen(function* () {
  const effectQuery = yield* ClientEffectQuery
  const queryClient = yield* QueryClient.QueryClient
  const runtime = yield* Effect.runtime<Registry.AtomRegistry>()
  const runFork = Runtime.runFork(runtime)

  /** Every read and resolution of host files rereads when any watched file changes. */
  const invalidateFiles = queryClient.invalidate(FilesBoundary.ReadFile.match()).pipe(
    Effect.zipRight(queryClient.invalidate(FilesBoundary.ResolvePath.match())),
  )

  /**
   * Open while observed: drains `WatchFile` for the file into invalidation, and
   * rereads on every (re)connection since events may have been missed.
   */
  const watch = memoized(
    (input: FileInput) => Key.canonical(input),
    (input: FileInput) => Atom.make((get): void => {
      const subscription = effectQuery.subscription(FilesBoundary.WatchFile, input)
      let attempt = 0
      get.subscribe(subscription, (state) => {
        if (state.attempt === attempt) return
        attempt = state.attempt
        if (attempt > 1) runFork(invalidateFiles)
      }, { immediate: true })
      const fiber = runFork(Subscription.events(subscription).pipe(
        Stream.runForEach(() => invalidateFiles),
      ))
      get.addFinalizer(() => {
        runFork(Fiber.interrupt(fiber))
      })
    }),
  )

  const resolve = memoized(
    (input: FileInput) => Key.canonical(input),
    (input: FileInput): Atom.Atom<Query.State<ResolvePathResult, Query.Error<typeof FilesBoundary.ResolvePath> | RpcClientError>> =>
      Atom.make((get) => {
        get(watch(input))
        return get(effectQuery.query(FilesBoundary.ResolvePath, { ...input, checkExists: true }))
      }),
  )

  const read = memoized(
    (input: FileInput & { readonly format: ReadFileFormat }) => Key.canonical(input),
    (input: FileInput & { readonly format: ReadFileFormat }): Atom.Atom<Query.State<ReadFileResult, Query.Error<typeof FilesBoundary.ReadFile> | RpcClientError>> =>
      Atom.make((get) => {
        get(watch({ cwd: input.cwd, path: input.path }))
        return get(effectQuery.query(FilesBoundary.ReadFile, input))
      }),
  )

  return {
    /** Resolution of `path` against `cwd`, live while observed. */
    resolve,
    /** Content of `path` under `cwd`, live while observed. */
    read,
  }
})

export interface Files extends Effect.Effect.Success<typeof makeFiles> {}

export const Files = Context.GenericTag<Files>("client/Files")

export const FilesLive = Layer.scoped(Files, makeFiles)
