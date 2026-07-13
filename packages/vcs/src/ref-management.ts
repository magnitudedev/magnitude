import { Effect } from "effect"
import type { VcsBackend } from "./backend"
import { VcsError, OperationNotFound } from "./errors"
import type { OperationId, CheckpointName, VcsCommit } from "./types"
import { parseCommitMessage } from "./commit-message"

/** Ref prefix for numbered checkpoints. */
const CHECKPOINT_PREFIX = "refs/checkpoints/"

/** Get the next checkpoint sequence number. */
export function nextCheckpointNumber(
  backend: VcsBackend,
): Effect.Effect<number, VcsError> {
  return Effect.gen(function* () {
    const refs = yield* backend.listRefs(CHECKPOINT_PREFIX).pipe(
      Effect.mapError((cause) => new VcsError({ operation: "listRefs", message: String(cause), cause })),
    )
    if (refs.length === 0) return 1
    const nums = refs
      .map((r) => {
        const raw = r.ref.slice(CHECKPOINT_PREFIX.length)
        // just-git may prefix the name with '/' after our prefix,
        // e.g. "refs/checkpoints//1" — strip the leading slash.
        const name = raw.startsWith("/") ? raw.slice(1) : raw
        const n = Number(name)
        return Number.isNaN(n) ? 0 : n
      })
      .filter((n) => n > 0)
    return nums.length === 0 ? 1 : Math.max(...nums) + 1
  })
}

/** Build a checkpoint ref name from a sequence number. */
export function checkpointRef(n: number): string {
  return `${CHECKPOINT_PREFIX}${n}`
}

/** Build a named checkpoint ref. */
export function namedCheckpointRef(name: string): string {
  return `${CHECKPOINT_PREFIX}${name}`
}

/** Read a checkpoint ref to get the commit hash. */
export function readCheckpointRef(
  backend: VcsBackend,
  name: string,
): Effect.Effect<string, OperationNotFound | VcsError> {
  return Effect.gen(function* () {
    const hash = yield* backend.readRef(namedCheckpointRef(name)).pipe(
      Effect.mapError((cause) => new VcsError({ operation: "readRef", message: String(cause), cause })),
    )
    if (hash === null) {
      return yield* new OperationNotFound({
        point: name,
        message: `Checkpoint '${name}' not found`,
      })
    }
    return hash
  })
}

/** List all checkpoint refs sorted by sequence number. */
export function listCheckpointRefs(
  backend: VcsBackend,
): Effect.Effect<ReadonlyArray<{ name: string; hash: string }>, VcsError> {
  return Effect.gen(function* () {
    const refs = yield* backend.listRefs(CHECKPOINT_PREFIX).pipe(
      Effect.mapError((cause) => new VcsError({ operation: "listRefs", message: String(cause), cause })),
    )
    const numeric = refs
      .map((r) => {
        const raw = r.ref.slice(CHECKPOINT_PREFIX.length)
        // just-git may prefix the name with '/' after our prefix,
        // e.g. "refs/checkpoints//1" — strip the leading slash.
        const name = raw.startsWith("/") ? raw.slice(1) : raw
        const n = Number(name)
        return { name, hash: r.hash, n: Number.isNaN(n) ? Infinity : n }
      })
      .sort((a, b) => a.n - b.n)
    return numeric.map((r) => ({ name: r.name, hash: r.hash }))
  })
}

/** Write a checkpoint ref. */
export function writeCheckpointRef(
  backend: VcsBackend,
  name: string,
  commitHash: string,
): Effect.Effect<void, VcsError> {
  return backend.updateRef(namedCheckpointRef(name), commitHash).pipe(
    Effect.mapError((cause) => new VcsError({ operation: "updateRef", message: String(cause), cause })),
  )
}

/** Delete a checkpoint ref. */
export function deleteCheckpointRef(
  backend: VcsBackend,
  name: string,
): Effect.Effect<void, VcsError> {
  return backend.deleteRef(namedCheckpointRef(name)).pipe(
    Effect.mapError((cause) => new VcsError({ operation: "deleteRef", message: String(cause), cause })),
  )
}

/** Read HEAD commit hash. */
export function readHead(
  backend: VcsBackend,
): Effect.Effect<string | null, VcsError> {
  return backend.readRef("HEAD").pipe(
    Effect.mapError((cause) => new VcsError({ operation: "readHead", message: String(cause), cause })),
  )
}

/** Update HEAD to a commit hash.
 *
 * HEAD is a symbolic ref pointing to refs/heads/main.
 * To "update HEAD", we update the branch it points to,
 * keeping HEAD as a symbolic ref.
 */
export function updateHead(
  backend: VcsBackend,
  commitHash: string,
): Effect.Effect<void, VcsError> {
  return backend.updateRef("refs/heads/main", commitHash).pipe(
    Effect.mapError((cause) => new VcsError({ operation: "updateHead", message: String(cause), cause })),
  )
}

/** Walk history and build VcsCommit entries.
 * Used by point-resolution and listing logic.
 *
 * NOTE: turn/agent/tool metadata are NOT stored in the commit message.
 * They must be resolved from the event stream at query time.
 */
export function walkCheckpointHistory(
  backend: VcsBackend,
  options?: { start?: string; limit?: number; pathFilter?: string },
): Effect.Effect<ReadonlyArray<VcsCommit>, VcsError> {
  return Effect.gen(function* () {
    const commits = yield* backend.walkHistory({
      start: options?.start,
      limit: options?.limit,
      pathFilter: options?.pathFilter,
    }).pipe(
      Effect.mapError((cause) => new VcsError({ operation: "walkHistory", message: String(cause), cause })),
    )

    const result: VcsCommit[] = []
    for (const c of commits) {
      const meta = parseCommitMessage(c.message)
      result.push({
        name: c.hash.slice(0, 7),
        operationId: c.hash as OperationId,
        commitHash: c.hash,
        treeHash: c.tree,
        timestamp: new Date(c.author.timestamp * 1000),
        message: meta.message,
      })
    }
    return result
  })
}
