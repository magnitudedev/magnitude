import { useCallback, useMemo } from "react"
import { Atom, Result, useAtomMount, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import * as Reactivity from "@effect/experimental/Reactivity"
import { Cause, Duration, Effect, Option, Schedule, Stream } from "effect"
import {
  type ProjectDirectoryListing,
  type ProjectEntryMove,
  type ProjectFileRevision,
  type ProjectFileSnapshot,
  type ProjectFileTextSnapshot,
  type ProjectId,
  type ProjectRelativePath,
} from "@magnitudedev/sdk"
import { useAgentClient } from "../state/agent-client-context"
import { visitProjectDirectoryDemand } from "./demand"

export interface ProjectPathInput {
  readonly projectId: ProjectId
  readonly path: ProjectRelativePath
}

export interface ProjectDirectoryInput {
  readonly projectId: ProjectId
  readonly directory: ProjectRelativePath
}

export interface ProjectFileWriteInput extends ProjectPathInput {
  readonly content: string
  readonly expectedRevision: ProjectFileRevision
}

export interface ProjectFileDeleteInput extends ProjectPathInput {
  readonly expectedRevision: ProjectFileRevision
}

export interface ProjectEntryMoveInput {
  readonly projectId: ProjectId
  readonly sourcePath: ProjectRelativePath
  readonly destinationDirectory: ProjectRelativePath
}

const projectFilesKey = (projectId: ProjectId): string => `project-files:${projectId}`
const projectFileKey = (projectId: ProjectId, path: ProjectRelativePath): string =>
  `${projectFilesKey(projectId)}:file:${path}`
const projectDirectoryKey = (projectId: ProjectId, path: ProjectRelativePath): string =>
  `${projectFilesKey(projectId)}:directory:${path}`
const parentDirectory = (path: ProjectRelativePath): ProjectRelativePath => {
  const index = path.lastIndexOf("/")
  return (index === -1 ? "" : path.slice(0, index)) as ProjectRelativePath
}
const initialFile = Atom.make(Result.initial<ProjectFileSnapshot>())
const directoryIdleTimeToLive = "2 minutes"
const projectFilesWatchReconnect = Schedule.exponential("100 millis").pipe(
  Schedule.modifyDelay((_, delay) => Duration.min(delay, Duration.seconds(5))),
  Schedule.jittered,
)

export function useProjectFilesWatch(projectId: ProjectId): void {
  const client = useAgentClient()
  const watchAtom = useMemo(() => client.rpc.runtime.atom(
    Effect.gen(function* () {
      const rpc = yield* client.rpc
      yield* Effect.gen(function* () {
        yield* Reactivity.invalidate([projectFilesKey(projectId)])
        yield* rpc("WatchProjectFiles", { projectId }).pipe(
          Stream.tap(() => Reactivity.invalidate([projectFilesKey(projectId)])),
          Stream.runDrain,
        )
      }).pipe(
        Effect.tapErrorCause((cause) => Cause.isInterruptedOnly(cause)
          ? Effect.void
          : Effect.logWarning("Project-files watch disconnected; retrying").pipe(
              Effect.annotateLogs({ projectId, cause: Cause.pretty(cause).slice(0, 1_000) }),
            )),
        Effect.retry(projectFilesWatchReconnect),
      )
    }).pipe(
      Effect.catchAllCause((cause) => Cause.isInterruptedOnly(cause)
        ? Effect.void
        : Effect.logError(Cause.pretty(cause))),
    ),
  ), [client, projectId])
  useAtomMount(watchAtom)
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
  root: ProjectRelativePath,
  demanded: ReadonlySet<ProjectRelativePath>,
  expanded: ReadonlySet<ProjectRelativePath>,
) {
  const client = useAgentClient()
  const demandedKey = [...demanded].sort().join("\0")
  const expandedKey = [...expanded].sort().join("\0")
  const tree = useMemo(() => Atom.make((get) => {
    const read = (directory: ProjectRelativePath) => get(client.rpc.query(
      "ListProjectDirectory",
      { projectId, directory },
      {
        timeToLive: directoryIdleTimeToLive,
        reactivityKeys: [
          projectFilesKey(projectId),
          projectDirectoryKey(projectId, directory),
        ],
      },
    ))
    const rootState = read(root)
    const rootListing = Result.value(rootState)
    const directories: Array<{
      readonly directory: ProjectRelativePath
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
  }), [client, projectId, root, demandedKey, expandedKey])
  return useAtomValue(tree)
}

export function useProjectDirectoryRefresh() {
  const client = useAgentClient()
  const refreshAtom = useMemo(() => client.rpc.runtime.fn<ProjectDirectoryInput>()(
    (input) => Reactivity.invalidate([projectDirectoryKey(input.projectId, input.directory)]),
  ), [client])
  return useAtomSet(refreshAtom)
}

export function useProjectFile(input: ProjectPathInput | null) {
  const client = useAgentClient()
  const query = useMemo(() => input === null
    ? initialFile
    : client.rpc.query("ReadProjectFile", input, {
        timeToLive: "0 millis",
        reactivityKeys: [
          projectFilesKey(input.projectId),
          projectFileKey(input.projectId, input.path),
        ],
      }), [client, input?.projectId, input?.path])
  return useAtomValue(query)
}

export function useProjectFileSave(options: {
  readonly onSuccess?: (snapshot: ProjectFileTextSnapshot) => void
} = {}) {
  const client = useAgentClient()
  const mutation = useMemo(() => client.rpc.mutation("WriteProjectFile"), [client])
  const result = useAtomValue(mutation)
  const execute = useAtomSet(mutation, { mode: "promise" })
  const save = useCallback(async (input: ProjectFileWriteInput) => {
    try {
      const snapshot = await execute({
        payload: input,
        reactivityKeys: [
          projectFileKey(input.projectId, input.path),
          projectDirectoryKey(input.projectId, parentDirectory(input.path)),
        ],
      })
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
  const mutation = useMemo(() => client.rpc.mutation("DeleteProjectFile"), [client])
  const result = useAtomValue(mutation)
  const execute = useAtomSet(mutation, { mode: "promise" })
  const remove = useCallback(async (input: ProjectFileDeleteInput) => {
    try {
      await execute({
        payload: input,
        reactivityKeys: [
          projectFileKey(input.projectId, input.path),
          projectDirectoryKey(input.projectId, parentDirectory(input.path)),
        ],
      })
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
  const mutation = useMemo(() => client.rpc.mutation("MoveProjectEntry"), [client])
  const result = useAtomValue(mutation)
  const execute = useAtomSet(mutation, { mode: "promise" })
  const move = useCallback(async (input: ProjectEntryMoveInput) => {
    try {
      const moved = await execute({
        payload: input,
        reactivityKeys: [projectFilesKey(input.projectId)],
      })
      options.onSuccess?.(moved)
    } catch {
      // The mutation Result is the rendered failure authority.
    }
  }, [execute, options.onSuccess])
  return { result, move }
}
