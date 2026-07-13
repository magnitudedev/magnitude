import { Layer, Effect } from "effect"
import type { ShadowVcsConfig } from "./types"
import { ShadowVcs } from "./service"
import { createJustGitBackend } from "./backends/just-git"
import type { VcsBackend } from "./backend"
import {
  VcsError,
  OperationNotFound,
  InvalidPointInTime,
  CorruptSnapshot,
  type VcsFailure,
} from "./errors"
import type { FileSystem } from "just-git"
import { VcsFs } from "./vcs-fs"

import type {
  OperationId,
  SnapshotId,
  CheckpointName,
  PointInTime,
  RestoreScope,
  Delta,
  VcsCommit,
  UndoRedoResult,
  RestoreResult,
} from "./types"
import {
  nextCheckpointNumber,
  namedCheckpointRef,
  readCheckpointRef,
  listCheckpointRefs,
  writeCheckpointRef,
  deleteCheckpointRef,
  readHead,
  updateHead,
} from "./ref-management"
import { parseCommitMessage } from "./commit-message"
import { createPathSelectorPredicate, filterDeltaBySelector } from "./path-selector"
import { createContentPatch } from "./diff-content"
import { getVcsToolEntries } from "./tools"

// ── Helper: map backend error to VcsError ────────────────────────────

function mapBackendErr(operation: string) {
  return (cause: unknown): VcsError => {
    const message = cause instanceof Error ? cause.message : String(cause)
    return new VcsError({ operation, message, cause })
  }
}

// ── Helper: read worktree file through backend ────────────────────

function readWorktreeFile(backend: VcsBackend, relativePath: string): Effect.Effect<Uint8Array | null, VcsError> {
  return backend.readWorktreeFile(relativePath).pipe(
    Effect.mapError(mapBackendErr("readWorktreeFile")),
  )
}

// (redo stack lives inside buildShadowVcs closure — per-instance)

// ── Helper: resolve a commit hash to its tree hash ──────────────────

function resolveTreeHash(
  backend: VcsBackend,
  commitHash: string,
): Effect.Effect<string, VcsFailure> {
  return Effect.gen(function* () {
    const commits = yield* backend.walkHistory({ start: commitHash, limit: 1 }).pipe(
      Effect.mapError(mapBackendErr("walkHistory")),
    )
    if (commits.length === 0) {
      return yield* new CorruptSnapshot({
        snapshotId: commitHash,
        message: `Commit ${commitHash} not found`,
      })
    }
    return commits[0]!.tree
  })
}

// ── Service implementation factory ────────────────────────────────────

export async function buildShadowVcs(backend: VcsBackend, worktreePath: string, timezone: string): Promise<ShadowVcs> {
  // ── Redo stack (per-instance) ────────────────────────────────────────
  const redoStack: string[] = []

  function clearRedoStack(): void {
    redoStack.length = 0
  }

  function pushRedo(commitHash: string): void {
    redoStack.push(commitHash)
  }

  function popRedoOne(): string | null {
    return redoStack.length > 0 ? redoStack.pop()! : null
  }

  function redoStackLength(): number {
    return redoStack.length
  }

  // Helper: get commit metadata from a commit hash
  const getCommitInfo = (
    hash: string,
    name: string,
  ): Effect.Effect<VcsCommit, VcsFailure> =>
    Effect.gen(function* () {
      const commits = yield* backend.walkHistory({ start: hash, limit: 1 }).pipe(
        Effect.mapError(mapBackendErr("walkHistory")),
      )
      if (commits.length === 0) {
        return yield* new CorruptSnapshot({
          snapshotId: hash,
          message: `Commit ${hash} not found`,
        })
      }
      const c = commits[0]!
      const meta = parseCommitMessage(c.message)
      const changedFiles = yield* (c.parents.length === 0
        // Initial commit: diff against empty tree
        ? backend.diffTree("", c.tree).pipe(
            Effect.map((d) => d.files.map((f) => f.path)),
            Effect.mapError(mapBackendErr("diffTree")),
          )
        // Non-initial commit: resolve parent's commit hash to its tree hash, then diff
        : Effect.gen(function* () {
            const parentTree = yield* resolveTreeHash(backend, c.parents[0]!)
            const delta = yield* backend.diffTree(parentTree, c.tree).pipe(
              Effect.mapError(mapBackendErr("diffTree")),
            )
            return delta.files.map((f) => f.path)
          }))
      return {
        name,
        operationId: hash as OperationId,
        commitHash: hash,
        treeHash: c.tree,
        timestamp: new Date(c.committer.timestamp * 1000),
        message: meta.message,
        filesChanged: changedFiles,
      }
    })

  // Resolve a PointInTime (or bare OperationId string) to a concrete OperationId (commit hash)
  const resolvePoint = (point: PointInTime | OperationId): Effect.Effect<OperationId, VcsFailure> =>
    Effect.gen(function* () {
      // Bare string — treat as operation ID
      if (typeof point === "string") {
        return point as OperationId
      }

      switch (point.kind) {
        case "operation":
          return point.id

        case "checkpoint":
          return (yield* readCheckpointRef(backend, point.name)) as OperationId

        case "snapshot": {
          // Walk all checkpoints, find the commit whose tree hash matches the snapshot id
          const all = yield* listCheckpointRefs(backend)
          for (const r of all) {
            const commits = yield* backend.walkHistory({ start: r.hash, limit: 1 }).pipe(
              Effect.orElse(() => Effect.succeed([] as ReadonlyArray<{ hash: string; message: string; tree: string; parents: ReadonlyArray<string>; author: { name: string; email: string; timestamp: number; timezone: string }; committer: { name: string; email: string; timestamp: number; timezone: string } }>)),
            )
            if (commits.length > 0 && commits[0]!.tree === point.id) {
              return r.hash as OperationId
            }
          }
          return yield* new OperationNotFound({
            point: `snapshot:${point.id}`,
            message: "No checkpoint found with matching tree hash",
          })
        }

        case "relative": {
          const anchor = yield* resolvePoint(point.anchor)
          const all = yield* listCheckpointRefs(backend)
          const idx = all.findIndex((r) => r.hash === anchor)
          if (idx === -1) {
            return yield* new InvalidPointInTime({
              point: `relative:${point.offset} from ${anchor}`,
              message: "Anchor not found in checkpoint list",
            })
          }
          const targetIdx = idx + point.offset
          if (targetIdx < 0 || targetIdx >= all.length) {
            return yield* new InvalidPointInTime({
              point: `relative:${point.offset} from ${anchor}`,
              message: "Relative offset out of bounds",
            })
          }
          return all[targetIdx]!.hash as OperationId
        }

        case "file": {
          const commits = yield* backend.walkHistory({ limit: 1000, pathFilter: point.path }).pipe(
            Effect.mapError(mapBackendErr("walkHistory")),
          )
          if (commits.length === 0) {
            return yield* new OperationNotFound({
              point: `file:${point.path}`,
              message: `No history found for ${point.path}`,
            })
          }
          return commits[0]!.hash as OperationId
        }

        case "time": {
          const commits = yield* backend.walkHistory({ limit: 1000 }).pipe(
            Effect.mapError(mapBackendErr("walkHistory")),
          )
          if (commits.length === 0) {
            return yield* new OperationNotFound({
              point: `time:${point.when.toISOString()}`,
              message: "No checkpoints exist",
            })
          }
          const targetMs = point.when.getTime()
          const candidates = commits.filter(c => c.committer.timestamp * 1000 <= targetMs)
          if (candidates.length === 0) {
            // No checkpoint before target time — fall back to the earliest commit
            // (walkHistory returns newest-first, so last element is oldest)
            const earliest = commits[commits.length - 1]!
            return earliest.hash as OperationId
          }
          // Find the maximum timestamp among candidates
          let maxTs = candidates[0]!.committer.timestamp
          for (const c of candidates) {
            if (c.committer.timestamp > maxTs) maxTs = c.committer.timestamp
          }
          // All commits at maxTs; return the oldest one (last in newest-first)
          const atMaxTs = candidates.filter(c => c.committer.timestamp === maxTs)
          const match = atMaxTs[atMaxTs.length - 1]!
          return match.hash as OperationId
        }

        case "message": {
          const commits = yield* backend.walkHistory({ limit: 1000 }).pipe(
            Effect.mapError(mapBackendErr("walkHistory")),
          )
          const match = commits.find((c) => c.message.includes(point.value))
          if (!match) {
            return yield* new OperationNotFound({
              point: `message:${point.value}`,
              message: "No commit matching message",
            })
          }
          return match.hash as OperationId
        }

        default:
          return yield* new InvalidPointInTime({
            point: String(point),
            message: "Unsupported PointInTime kind",
          })
      }
    })

  // Helper: build files map from getChangedFiles result
  // (uses top-level buildFilesMap function)

  return {
    timezone,

    getTools: getVcsToolEntries,

    shutdown: Effect.sync(() => {
      clearRedoStack()
    }),

    record: (options) =>
      Effect.gen(function* () {
        const changes = yield* backend.getChangedFiles().pipe(
          Effect.mapError(mapBackendErr("getChangedFiles")),
        )

        if (changes.length === 0) {
          const headHash = yield* backend.readRef("HEAD").pipe(
            Effect.mapError(mapBackendErr("readRef")),
          )
          if (headHash === null) {
            return yield* new VcsError({
              operation: "record",
              message: "No changes and HEAD is missing — repo not initialised?",
            })
          }
          return headHash as OperationId
        }

        const commitHash = yield* backend.buildCommit({
          message: options?.message ?? "",
        }).pipe(
          Effect.mapError(mapBackendErr("buildCommit")),
        )

        const nextNum = yield* nextCheckpointNumber(backend)
        yield* writeCheckpointRef(backend, String(nextNum), commitHash)

        clearRedoStack()

        return commitHash as OperationId
      }),

    head: Effect.gen(function* () {
      const headHash = yield* backend.readRef("HEAD").pipe(
        Effect.mapError(mapBackendErr("readRef")),
      )
      if (headHash === null) {
        return yield* new OperationNotFound({
          point: "HEAD",
          message: "No HEAD ref",
        })
      }
      const all = yield* listCheckpointRefs(backend)
      const name = all.find((r) => r.hash === headHash)?.name ?? "HEAD"
      return yield* getCommitInfo(headHash, name)
    }),

    resolve: (point) => resolvePoint(point),

    getCheckpoint: (nameOrId) =>
      Effect.gen(function* () {
        // Try as a ref first
        const asRef = yield* backend.readRef(namedCheckpointRef(nameOrId)).pipe(
          Effect.mapError(() => null),
          Effect.orElse(() => Effect.succeed(null)),
        )
        if (asRef) return yield* getCommitInfo(asRef, nameOrId)

        // Try as a raw hash
        const commits = yield* backend.walkHistory({ start: nameOrId, limit: 1 }).pipe(
          Effect.orElse(() => Effect.succeed([] as ReadonlyArray<{ hash: string; message: string; tree: string; parents: ReadonlyArray<string>; author: { name: string; email: string; timestamp: number; timezone: string }; committer: { name: string; email: string; timestamp: number; timezone: string } }>)),
        )
        if (commits.length > 0) return yield* getCommitInfo(nameOrId, nameOrId)

        return yield* new OperationNotFound({
          point: nameOrId,
          message: `No checkpoint found for '${nameOrId}'`,
        })
      }),

    listCheckpoints: (options) =>
      Effect.gen(function* () {
        const refs = yield* listCheckpointRefs(backend)
        let filtered = refs
        if (options?.from) {
          const fromId = yield* resolvePoint(options.from)
          const idx = refs.findIndex((r) => r.hash === fromId)
          if (idx !== -1) filtered = filtered.slice(idx)
        }
        if (options?.to) {
          const toId = yield* resolvePoint(options.to)
          const idx = refs.findIndex((r) => r.hash === toId)
          if (idx !== -1) filtered = filtered.slice(0, idx + 1)
        }
        const limit = options?.limit ?? 100
        filtered = filtered.slice(-limit)

        const infos: VcsCommit[] = []
        for (const r of filtered) {
          const info = yield* getCommitInfo(r.hash, r.name)
          infos.push(info)
        }
        return infos
      }),

    diff: (options) =>
      Effect.gen(function* () {
        const fromId = yield* resolvePoint(options.from)
        const toId = yield* resolvePoint(options.to)
        const fromCommits = yield* backend.walkHistory({ start: fromId, limit: 1 }).pipe(
          Effect.mapError(mapBackendErr("walkHistory")),
        )
        const toCommits = yield* backend.walkHistory({ start: toId, limit: 1 }).pipe(
          Effect.mapError(mapBackendErr("walkHistory")),
        )
        if (fromCommits.length === 0 || toCommits.length === 0) {
          return yield* new OperationNotFound({
            point: `${fromId}..${toId}`,
            message: "Cannot resolve diff endpoints",
          })
        }
        const delta = yield* backend.diffTree(
          fromCommits[0]!.tree,
          toCommits[0]!.tree,
        ).pipe(
          Effect.mapError(mapBackendErr("diffTree")),
        )

        return filterDeltaBySelector(delta, options.pathFilter)
      }),

    diffWorking: (options) =>
      Effect.gen(function* () {
        const againstId = yield* resolvePoint(options.against)

        // Resolve the 'against' commit to its tree hash
        const againstCommits = yield* backend.walkHistory({ start: againstId, limit: 1 }).pipe(
          Effect.mapError(mapBackendErr("walkHistory")),
        )
        if (againstCommits.length === 0) {
          return yield* new OperationNotFound({
            point: againstId,
            message: "Cannot resolve diff endpoint",
          })
        }
        const againstTree = againstCommits[0]!.tree

        // diffWorking compares worktree state against a checkpoint.
        // Two cases:
        // 1. Worktree is clean vs HEAD → diff againstTree vs HEAD's tree
        // 2. Worktree is dirty → diff againstTree vs worktree content
        const worktreeChanges = yield* backend.getChangedFiles().pipe(
          Effect.mapError(mapBackendErr("getChangedFiles")),
        )

        if (worktreeChanges.length === 0) {
          // Worktree matches HEAD — diff the two committed trees directly
          const headHash = yield* backend.readRef("HEAD").pipe(
            Effect.mapError(mapBackendErr("readRef")),
          )
          if (headHash === null) {
            return { additions: 0, deletions: 0, modifications: 0, renames: 0, files: [] }
          }
          const headCommits = yield* backend.walkHistory({ start: headHash, limit: 1 }).pipe(
            Effect.mapError(mapBackendErr("walkHistory")),
          )
          if (headCommits.length === 0) {
            return { additions: 0, deletions: 0, modifications: 0, renames: 0, files: [] }
          }
          const headTree = headCommits[0]!.tree
          const delta = yield* backend.diffTree(againstTree, headTree).pipe(
            Effect.mapError(mapBackendErr("diffTree")),
          )
          return filterDeltaBySelector(delta, options.pathFilter)
        }

        // Worktree is dirty — show worktree changes vs the 'against' checkpoint
        let changedFiles = worktreeChanges
        if (options.pathFilter) {
          const matches = createPathSelectorPredicate(options.pathFilter)
          changedFiles = changedFiles.filter(cf => matches(cf.path))
        }

        // Batch read all old and new content
        const oldContents: Array<Uint8Array | null> = yield* Effect.all(
          changedFiles.map(cf =>
            cf.status !== "added"
              ? backend.readFileAt(againstTree, cf.path).pipe(
                  Effect.mapError(mapBackendErr("readFileAt")),
                  Effect.orElse(() => Effect.succeed(null)),
                )
              : Effect.succeed(null),
          ),
        )
        const newContents: Array<Uint8Array | null> = yield* Effect.all(
          changedFiles.map(cf =>
            cf.status !== "deleted"
              ? backend.readWorktreeFile(cf.path).pipe(
                  Effect.mapError(mapBackendErr("readWorktreeFile")),
                  Effect.orElse(() => Effect.succeed(null)),
                )
              : Effect.succeed(null),
          ),
        )

        let additions = 0
        let deletions = 0
        let modifications = 0
        let renames = 0
        const files: Array<{
          path: string
          status: "added" | "deleted" | "modified" | "renamed"
          oldPath?: string
          diff: string
        }> = []

        for (let i = 0; i < changedFiles.length; i++) {
          const cf = changedFiles[i]!
          const oldBytes = oldContents[i]
          const newBytes = newContents[i]
          const patch = createContentPatch(cf.path, oldBytes ?? null, newBytes ?? null)

          if (cf.status === "added") additions++
          else if (cf.status === "deleted") deletions++
          else if (cf.status === "modified") modifications++

          files.push({ path: cf.path, status: cf.status, diff: patch })
        }

        return {
          additions,
          deletions,
          modifications,
          renames,
          files,
        }
      }),

    restore: (options) =>
      Effect.gen(function* () {
        const targetId = yield* resolvePoint(options.to)
        clearRedoStack()

        // Resolve the target commit's tree hash for the return value
        const targetCommits = yield* backend.walkHistory({ start: targetId, limit: 1 }).pipe(
          Effect.mapError(mapBackendErr("walkHistory")),
        )
        if (targetCommits.length === 0) {
          return yield* new OperationNotFound({
            point: targetId,
            message: "Cannot resolve target commit for restore",
          })
        }
        const targetTreeHash = targetCommits[0]!.tree as SnapshotId

        // 1. Auto-save current state BEFORE mutating anything
        const preChanges = yield* backend.getChangedFiles().pipe(
          Effect.mapError(mapBackendErr("getChangedFiles")),
        )
        let preCommit: string
        if (preChanges.length === 0) {
          const headHash = yield* backend.readRef("HEAD").pipe(
            Effect.mapError(mapBackendErr("readRef")),
          )
          if (headHash === null) {
            return yield* new VcsError({
              operation: "restore",
              message: "Cannot restore — repo has no HEAD",
            })
          }
          preCommit = headHash
        } else {
          preCommit = yield* backend.buildCommit({
            message: `pre-restore-${targetId.slice(0, 7)}`,
          }).pipe(
            Effect.mapError(mapBackendErr("buildCommit")),
          )
        }
        const preNum = yield* nextCheckpointNumber(backend)
        yield* writeCheckpointRef(backend, `pre-restore-${preNum}`, preCommit)

        // 2. Extract target commit to worktree (handles writes + deletions)
        yield* backend.extractTree(targetId, options.scope).pipe(
          Effect.mapError(mapBackendErr("extractTree")),
        )

        // 3. Record post-restore checkpoint — capture actual worktree state
        //    (after extractTree, worktree matches the target tree)
        const postChanges = yield* backend.getChangedFiles().pipe(
          Effect.mapError(mapBackendErr("getChangedFiles")),
        )
        let postCommit: string
        if (postChanges.length === 0) {
          // Worktree matches HEAD — no new commit needed
          const headHash = yield* backend.readRef("HEAD").pipe(
            Effect.mapError(mapBackendErr("readRef")),
          )
          postCommit = headHash ?? preCommit
        } else {
          postCommit = yield* backend.buildCommit({
            message: `post-restore-${targetId.slice(0, 7)}`,
          }).pipe(
            Effect.mapError(mapBackendErr("buildCommit")),
          )
        }
        const postNum = yield* nextCheckpointNumber(backend)
        yield* writeCheckpointRef(backend, `post-restore-${postNum}`, postCommit)

        return {
          targetSnapshotId: targetTreeHash,
          undoOperationId: preCommit as OperationId,
        }
      }),

    undo: (options) =>
      Effect.gen(function* () {
        const headHash = yield* backend.readRef("HEAD").pipe(
          Effect.mapError(mapBackendErr("readRef")),
        )
        if (headHash === null) {
          return yield* new OperationNotFound({ point: "HEAD", message: "No HEAD to undo" })
        }

        const commits = yield* backend.walkHistory({ start: headHash, limit: 1 }).pipe(
          Effect.mapError(mapBackendErr("walkHistory")),
        )
        if (commits.length === 0 || commits[0]!.parents.length === 0) {
          return yield* new OperationNotFound({
            point: "HEAD",
            message: "No parent to undo to",
          })
        }
        const parentHash = commits[0]!.parents[0]!

        // 1. Auto-save current state
        const preChanges = yield* backend.getChangedFiles().pipe(
          Effect.mapError(mapBackendErr("getChangedFiles")),
        )
        let preCommit: string
        if (preChanges.length === 0) {
          preCommit = headHash
        } else {
          preCommit = yield* backend.buildCommit({
            message: "pre-undo-auto",
          }).pipe(
            Effect.mapError(mapBackendErr("buildCommit")),
          )
        }
        const preNum = yield* nextCheckpointNumber(backend)
        yield* writeCheckpointRef(backend, `pre-undo-${preNum}`, preCommit)

        // 2. Push old HEAD onto redo stack
        pushRedo(headHash)

        // 3. Move HEAD to parent
        yield* updateHead(backend, parentHash)

        // 4. Restore worktree to parent commit
        yield* backend.extractTree(parentHash, options?.scope).pipe(
          Effect.mapError(mapBackendErr("extractTree")),
        )

        return {
          restoredOperationId: parentHash as OperationId,
          undoOperationId: preCommit as OperationId,
        }
      }),

    redo: (options) =>
      Effect.gen(function* () {
        if (redoStackLength() === 0) {
          return yield* new OperationNotFound({
            point: "redo",
            message: "No redo operations available",
          })
        }
        const targetHash = popRedoOne()!

        // 1. Auto-save current state
        const preChanges = yield* backend.getChangedFiles().pipe(
          Effect.mapError(mapBackendErr("getChangedFiles")),
        )
        let preCommit: string
        if (preChanges.length === 0) {
          preCommit = targetHash
        } else {
          preCommit = yield* backend.buildCommit({
            message: "pre-redo-auto",
          }).pipe(
            Effect.mapError(mapBackendErr("buildCommit")),
          )
        }
        const preNum = yield* nextCheckpointNumber(backend)
        yield* writeCheckpointRef(backend, `pre-redo-${preNum}`, preCommit)

        // 2. Move HEAD to redo target
        yield* updateHead(backend, targetHash)

        // 3. Restore worktree to redo target
        yield* backend.extractTree(targetHash, options?.scope).pipe(
          Effect.mapError(mapBackendErr("extractTree")),
        )

        return {
          restoredOperationId: targetHash as OperationId,
          undoOperationId: preCommit as OperationId,
        }
      }),

    readAt: (options) =>
      Effect.gen(function* () {
        const id = yield* resolvePoint(options.point)
        const commits = yield* backend.walkHistory({ start: id, limit: 1 }).pipe(
          Effect.mapError(mapBackendErr("walkHistory")),
        )
        if (commits.length === 0) {
          return yield* new OperationNotFound({
            point: id,
            message: "Point not found",
          })
        }
        const treeHash = commits[0]!.tree
        const result = new Map<string, Uint8Array>()
        for (const p of options.paths) {
          const bytes = yield* backend.readFileAt(treeHash, p).pipe(
            Effect.mapError(mapBackendErr("readFileAt")),
          )
          if (bytes !== null) {
            result.set(p, bytes)
          }
        }
        return result
      }),

    checkpoint: (options) =>
      Effect.gen(function* () {
        const atId = options.at
          ? yield* resolvePoint(options.at)
          : yield* Effect.gen(function* () {
              const head = yield* backend.readRef("HEAD").pipe(
                Effect.mapError(mapBackendErr("readRef")),
              )
              if (head === null) {
                return yield* new OperationNotFound({
                  point: "HEAD",
                  message: "No HEAD to bookmark",
                })
              }
              return head as OperationId
            })
        yield* writeCheckpointRef(backend, options.name, atId)
      }),

    deleteCheckpoint: (name) =>
      deleteCheckpointRef(backend, name).pipe(
        Effect.catchAll(() => Effect.void),
      ),

    listNamedCheckpoints: () =>
      Effect.gen(function* () {
        const refs = yield* listCheckpointRefs(backend)
        const named = refs.filter((r) => Number.isNaN(Number(r.name)))
        return named.map((r) => ({
          name: r.name,
          operationId: r.hash as OperationId,
        }))
      }),

    historyForPath: (options) =>
      Effect.gen(function* () {
        const commits = yield* backend.walkHistory({
          limit: options.limit ?? 100,
          pathFilter: options.path,
        }).pipe(
          Effect.mapError(mapBackendErr("walkHistory")),
        )
        const infos: VcsCommit[] = []
        for (const c of commits) {
          const meta = parseCommitMessage(c.message)
          const delta = yield* (c.parents.length === 0
            ? backend.diffTree("", c.tree).pipe(
                Effect.map((delta) => filterDeltaBySelector(delta, options.path)),
                Effect.mapError(mapBackendErr("diffTree")),
              )
            : Effect.gen(function* () {
                const parentTree = yield* resolveTreeHash(backend, c.parents[0]!)
                const d = yield* backend.diffTree(parentTree, c.tree).pipe(
                  Effect.map((delta) => filterDeltaBySelector(delta, options.path)),
                  Effect.mapError(mapBackendErr("diffTree")),
                )
                return d
              }))
          infos.push({
            name: c.hash.slice(0, 7),
            operationId: c.hash as OperationId,
            commitHash: c.hash,
            treeHash: c.tree,
            timestamp: new Date(c.committer.timestamp * 1000),
            message: meta.message,
            filesChanged: delta.files.map((f) => f.path),
          })
        }
        return infos
      }),

    isClean: Effect.gen(function* () {
      const changes = yield* backend.getChangedFiles().pipe(
        Effect.mapError(mapBackendErr("getChangedFiles")),
      )
      return changes.length === 0
    }),

    changedSinceHead: Effect.gen(function* () {
      const changes = yield* backend.getChangedFiles().pipe(
        Effect.mapError(mapBackendErr("getChangedFiles")),
      )
      return changes.map((c) => c.path)
    }),
  }
}

// ── Full factory: create backend + run initial checkpoint + build service ──

async function makeShadowVcs(config: ShadowVcsConfig, fs: FileSystem): Promise<ShadowVcs> {
  const worktreePath = config.worktreePath
  const gitDirPath = config.storagePath + "/.git"

  const backend: VcsBackend = await createJustGitBackend(worktreePath, gitDirPath, fs)

  // Check if repo already initialized (HEAD exists and resolves)
  const headExists = await Effect.runPromise(
    Effect.gen(function* () {
      const head = yield* backend.readRef("HEAD").pipe(
        Effect.mapError(mapBackendErr("readRef")),
      )
      return head !== null
    }).pipe(Effect.catchAll(() => Effect.succeed(false))),
  )

  if (!headExists) {
    // Run initial checkpoint
    await Effect.runPromise(
      Effect.gen(function* () {
        const changes = yield* backend.getChangedFiles().pipe(
          Effect.mapError(mapBackendErr("getChangedFiles")),
        )
        if (changes.length > 0) {
          const commitHash = yield* backend.buildCommit({
            message: "initial checkpoint",
          }).pipe(
            Effect.mapError(mapBackendErr("buildCommit")),
          )
          yield* writeCheckpointRef(backend, "1", commitHash)
        } else {
          // Empty worktree — create empty commit
          const commitHash = yield* backend.buildCommit({
            message: "initial checkpoint",
          }).pipe(
            Effect.mapError(mapBackendErr("buildCommit")),
          )
          yield* writeCheckpointRef(backend, "1", commitHash)
        }
      }),
    )
  }

  return buildShadowVcs(backend, worktreePath, config.timezone)
}

// ── Layer factory ───────────────────────────────────────────────────

/** Create an Effect Layer that provides ShadowVcs for the given config.
 *  Requires VcsFs in the Effect context — provides the FileSystem
 *  to use for worktree operations. */
export function makeShadowVcsLayer(config: ShadowVcsConfig): Layer.Layer<ShadowVcs, never, VcsFs> {
  return Layer.effect(ShadowVcs, Effect.gen(function* () {
    const fs = yield* VcsFs
    return yield* Effect.promise(() => makeShadowVcs(config, fs))
  }))
}
