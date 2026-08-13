import { useMemo } from "react"
import { Equivalence, Option, Schema } from "effect"
import { Atom, Result, useAtomValue } from "@effect-atom/atom-react"
import type * as Rpc from "@effect/rpc/Rpc"
import type * as RpcGroup from "@effect/rpc/RpcGroup"
import type { RpcClientError } from "@effect/rpc/RpcClientError"
import type { MagnitudeRpcs } from "@magnitudedev/sdk"
import { useAgentClient } from "../state/agent-client-context"

type MagnitudeRpc = RpcGroup.Rpcs<typeof MagnitudeRpcs>
type RpcPayload<Tag extends Rpc.Tag<MagnitudeRpc>> = Rpc.PayloadConstructor<Rpc.ExtractTag<MagnitudeRpc, Tag>>
/**
 * Mirrors one protocol-defined backend state into a query atom. The definition's
 * RPC tag is also its invalidation identity, so no parallel client configuration exists.
 */
export function useMirroredState<
  const Id extends Rpc.Tag<MagnitudeRpc>,
  Snapshot,
  SnapshotEncoded,
  SnapshotRequirements,
  Error,
  ErrorEncoded,
  ErrorRequirements,
>(definition: {
  readonly id: Id
  readonly getPayload: RpcPayload<Id>
  readonly snapshotSchema: Schema.Schema<Snapshot, SnapshotEncoded, SnapshotRequirements>
  readonly errorSchema: Schema.Schema<Error, ErrorEncoded, ErrorRequirements>
}): Result.Result<Snapshot, Error | RpcClientError> {
  const queryAtom = useMirroredStateAtom(definition)
  return useAtomValue(queryAtom)
}

/**
 * Selects a semantically stable value from one mirrored-state query.
 *
 * The selector remains a read-only projection of the canonical query atom. Its
 * retained value is only an equality cache: when a fresh snapshot projects to
 * an equivalent value, consumers keep the previous reference and do not
 * rerender. No server state is copied into a writable client atom.
 */
export function useMirroredStateSelector<
  const Id extends Rpc.Tag<MagnitudeRpc>,
  Snapshot,
  SnapshotEncoded,
  SnapshotRequirements,
  Error,
  ErrorEncoded,
  ErrorRequirements,
  Selection,
>(
  definition: {
    readonly id: Id
    readonly getPayload: RpcPayload<Id>
    readonly snapshotSchema: Schema.Schema<Snapshot, SnapshotEncoded, SnapshotRequirements>
    readonly errorSchema: Schema.Schema<Error, ErrorEncoded, ErrorRequirements>
  },
  selector: (snapshot: Snapshot) => Selection,
  equivalent: Equivalence.Equivalence<Selection>,
): Option.Option<Selection> {
  const queryAtom = useMirroredStateAtom(definition)
  const selectorAtom = useMemo(() => {
    let previousSelection = Option.none<Selection>()
    return Atom.make((get) => Option.map(
      Result.value(get(queryAtom)),
      (snapshot) => {
        const nextSelection = selector(snapshot)
        if (Option.isSome(previousSelection)
          && equivalent(previousSelection.value, nextSelection)) {
          return previousSelection.value
        }
        previousSelection = Option.some(nextSelection)
        return nextSelection
      },
    ))
  }, [equivalent, queryAtom, selector])
  return useAtomValue(selectorAtom)
}

/**
 * Returns the query atom for one directly mirrored domain. The connection
 * runtime owns invalidation; observing this atom only observes the snapshot.
 */
export function useMirroredStateAtom<
  const Id extends Rpc.Tag<MagnitudeRpc>,
  Snapshot,
  SnapshotEncoded,
  SnapshotRequirements,
  Error,
  ErrorEncoded,
  ErrorRequirements,
>(definition: {
  readonly id: Id
  readonly getPayload: RpcPayload<Id>
  readonly snapshotSchema: Schema.Schema<Snapshot, SnapshotEncoded, SnapshotRequirements>
  readonly errorSchema: Schema.Schema<Error, ErrorEncoded, ErrorRequirements>
}): Atom.Atom<Result.Result<Snapshot, Error | RpcClientError>> {
  const client = useAgentClient()
  const queryAtom = useMemo(
    () =>
      client.rpc.query(definition.id, definition.getPayload, {
        reactivityKeys: [definition.id],
        timeToLive: Infinity,
      }),
    [client, definition],
  )
  return queryAtom
}
