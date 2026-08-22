/**
 * Project files: directory listings and file snapshots kept fresh by
 * `WatchProjectFiles`. Freshness is a dependency of observation — a project's
 * watch is open exactly while one of its listings or files is observed.
 * Writes, deletions, and moves are contract mutations whose postconditions
 * invalidate the affected queries.
 */
import { useCallback, useMemo } from "react"
import type { RpcClientError } from "@effect/rpc/RpcClientError"
import { Atom, Registry, Result, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { Context, Effect, Fiber, Layer, Option, Runtime, Stream } from "effect"
import { Key, QueryClient, Subscription, type Query } from "@magnitudedev/effect-query"
import {
  ProjectFiles as ProjectFilesBoundary,
  type FileContentHash,
  type ProjectDirectoryListing,
  type ProjectEntryMove,
  type ProjectFileSnapshot,
  type ProjectFileTextSnapshot,
  type ProjectId,
  type RelativePath,
} from "@magnitudedev/sdk"
import { useAgentClient } from "../state/agent-client-context"
import { ClientEffectQuery } from "../state/client-effect-query"
import { visitProjectDirectoryDemand } from "./demand"

export interface ProjectPathInput {
  readonly projectId: ProjectId
  readonly path: RelativePath
}

export interface ProjectDirectoryInput {
  readonly projectId: ProjectId
  readonly directory: RelativePath
}

export interface ProjectFileWriteInput extends ProjectPathInput {
  readonly content: string
  readonly expectedContentHash: FileContentHash
}

export interface ProjectFileDeleteInput extends ProjectPathInput {
  readonly expectedContentHash: FileContentHash
}

export interface ProjectEntryMoveInput {
  readonly projectId: ProjectId
  readonly sourcePath: RelativePath
  readonly destinationDirectory: RelativePath
}

type DirectoryState = Query.State<ProjectDirectoryListing, Query.Error<typeof ProjectFilesBoundary.ListProjectDirectory> | RpcClientError>
type FileState = Query.State<ProjectFileSnapshot, Query.Error<typeof ProjectFilesBoundary.ReadProjectFile> | RpcClientError>

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

const makeProjectFiles = Effect.gen(function* () {
  const effectQuery = yield* ClientEffectQuery
  const queryClient = yield* QueryClient.QueryClient
  const runtime = yield* Effect.runtime<Registry.AtomRegistry>()
  const runFork = Runtime.runFork(runtime)

  /** A change notification rereads every observed listing and file (one project is open at a time). */
  const invalidateProject = queryClient.invalidate(ProjectFilesBoundary.ListProjectDirectory.match()).pipe(
    Effect.zipRight(queryClient.invalidate(ProjectFilesBoundary.ReadProjectFile.match())),
  )

  /**
   * Open while observed: drains `WatchProjectFiles` for the project into
   * invalidation, and rereads on every (re)connection since events may have
   * been missed.
   */
  const watch = memoized(
    (projectId: ProjectId) => projectId,
    (projectId: ProjectId) => Atom.make((get): void => {
      const subscription = effectQuery.subscription(ProjectFilesBoundary.WatchProjectFiles, { projectId })
      let attempt = 0
      get.subscribe(subscription, (state) => {
        if (state.attempt === attempt) return
        attempt = state.attempt
        if (attempt > 1) runFork(invalidateProject)
      }, { immediate: true })
      const fiber = runFork(Subscription.events(subscription).pipe(
        Stream.runForEach(() => invalidateProject),
      ))
      get.addFinalizer(() => {
        runFork(Fiber.interrupt(fiber))
      })
    }),
  )

  const directory = memoized(
    (input: ProjectDirectoryInput) => Key.canonical(input),
    (input: ProjectDirectoryInput): Atom.Atom<DirectoryState> => Atom.make((get) => {
      get(watch(input.projectId))
      return get(effectQuery.query(ProjectFilesBoundary.ListProjectDirectory, input))
    }),
  )

  const file = memoized(
    (input: ProjectPathInput) => Key.canonical(input),
    (input: ProjectPathInput): Atom.Atom<FileState> => Atom.make((get) => {
      get(watch(input.projectId))
      return get(effectQuery.query(ProjectFilesBoundary.ReadProjectFile, input))
    }),
  )

  return {
    /** One directory listing, live while observed. */
    directory,
    /** One file snapshot, live while observed. */
    file,
    /** Rereads one directory listing now. */
    refresh: (input: ProjectDirectoryInput) => queryClient.invalidate(ProjectFilesBoundary.ListProjectDirectory.match(input)),
  }
})

export interface ProjectFiles extends Effect.Effect.Success<typeof makeProjectFiles> {}

export const ProjectFiles = Context.GenericTag<ProjectFiles>("client/ProjectFiles")

export const ProjectFilesLive = Layer.scoped(ProjectFiles, makeProjectFiles)

const initialFile: Result.Result<ProjectFileSnapshot, unknown> = Result.initial()
const initialDirectory: Result.Result<ProjectDirectoryListing, unknown> = Result.initial()

export interface ProjectDirectoryTree {
  readonly root: Result.Result<ProjectDirectoryListing, unknown>
  readonly directories: ReadonlyArray<{
    readonly directory: RelativePath
    readonly state: Result.Result<ProjectDirectoryListing, unknown>
  }>
}

/**
 * Materialize only the demanded portion of a project tree.
 *
 * The root query is observed first. Descendant queries become dependencies only
 * after their authoritative parent listing exists, so restored expansion walks
 * breadth-first and stale deep paths never issue speculative RPCs.
 */
export function useProjectDirectoryTree(
  projectId: ProjectId,
  root: RelativePath,
  demanded: ReadonlySet<RelativePath>,
  expanded: ReadonlySet<RelativePath>,
): ProjectDirectoryTree {
  const client = useAgentClient()
  const service = useMemo(() => client.runtime.atom(ProjectFiles), [client])
  const demandedKey = [...demanded].sort().join("\0")
  const expandedKey = [...expanded].sort().join("\0")
  const tree = useMemo(() => Atom.make((get): ProjectDirectoryTree => {
    const files = Result.value(get(service))
    if (Option.isNone(files)) return { root: initialDirectory, directories: [] }
    const read = (directory: RelativePath) => get(files.value.directory({ projectId, directory })).result
    const rootState = read(root)
    const rootListing = Result.value(rootState)
    const directories: Array<{
      readonly directory: RelativePath
      readonly state: typeof rootState
    }> = []
    if (Option.isNone(rootListing)) return { root: rootState, directories }

    visitProjectDirectoryDemand(rootListing.value.entries, demanded, (directory) => {
      const state = read(directory)
      directories.push({ directory, state })
      if (!expanded.has(directory)) return undefined
      const listing = Result.value(state)
      return Option.isSome(listing) ? listing.value.entries : undefined
    })

    return { root: rootState, directories }
  }), [service, projectId, root, demandedKey, expandedKey])
  return useAtomValue(tree)
}

export function useProjectDirectoryRefresh() {
  const client = useAgentClient()
  const refreshAtom = useMemo(() => client.runtime.fn<ProjectDirectoryInput>()(
    (input) => Effect.flatMap(ProjectFiles, (files) => files.refresh(input)),
  ), [client])
  return useAtomSet(refreshAtom)
}

export function useProjectFile(input: ProjectPathInput | null) {
  const client = useAgentClient()
  const service = useMemo(() => client.runtime.atom(ProjectFiles), [client])
  const query = useMemo(() => Atom.make((get): Result.Result<ProjectFileSnapshot, unknown> => {
    if (input === null) return initialFile
    const files = Result.value(get(service))
    if (Option.isNone(files)) return initialFile
    return get(files.value.file(input)).result
  }), [service, input?.projectId, input?.path])
  return useAtomValue(query)
}

export function useProjectFileSave(options: {
  readonly onSuccess?: (snapshot: ProjectFileTextSnapshot) => void
} = {}) {
  const client = useAgentClient()
  const mutation = useMemo(() => client.mutation(ProjectFilesBoundary.WriteProjectFile), [client])
  const result = useAtomValue(mutation)
  const execute = useAtomSet(mutation, { mode: "promise" })
  const save = useCallback(async (input: ProjectFileWriteInput) => {
    try {
      const snapshot = await execute(input)
      options.onSuccess?.(snapshot)
    } catch {
      // The mutation Result is the rendered failure authority.
    }
  }, [execute, options.onSuccess])
  return { result, save }
}

export function useProjectFileDelete(options: {
  readonly onSuccess?: () => void
} = {}) {
  const client = useAgentClient()
  const mutation = useMemo(() => client.mutation(ProjectFilesBoundary.DeleteProjectFile), [client])
  const result = useAtomValue(mutation)
  const execute = useAtomSet(mutation, { mode: "promise" })
  const remove = useCallback(async (input: ProjectFileDeleteInput) => {
    try {
      await execute(input)
      options.onSuccess?.()
    } catch {
      // The mutation Result is the rendered failure authority.
    }
  }, [execute, options.onSuccess])
  return { result, remove }
}

export function useProjectEntryMove(options: {
  readonly onSuccess?: (move: ProjectEntryMove) => void
} = {}) {
  const client = useAgentClient()
  const mutation = useMemo(() => client.mutation(ProjectFilesBoundary.MoveProjectEntry), [client])
  const result = useAtomValue(mutation)
  const execute = useAtomSet(mutation, { mode: "promise" })
  const move = useCallback(async (input: ProjectEntryMoveInput) => {
    try {
      const moved = await execute(input)
      options.onSuccess?.(moved)
    } catch {
      // The mutation Result is the rendered failure authority.
    }
  }, [execute, options.onSuccess])
  return { result, move }
}
