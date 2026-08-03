import { useMemo, useRef, useSyncExternalStore } from 'react'
import { Atom, useAtomMount } from '@effect-atom/atom-react'
import { Clock, Effect } from 'effect'
import type { DisplayRootStatus } from '@magnitudedev/sdk'

type WorkingStatus = Extract<DisplayRootStatus, { readonly _tag: 'Working' }>
type DisplayRootDetail = WorkingStatus['detail']

interface DetailInput {
  readonly chainStartedAt: number
  readonly detail: DisplayRootDetail
}

interface DetailSnapshot {
  readonly chainStartedAt: number
  readonly key: string
}

const DEFAULT_SETTLE_MS = 150
const WAITING_FOR_MODEL_DELAY_MS = 1_000
const NO_DETAIL: DisplayRootDetail = { _tag: 'NoDetail' }

function detailKey(detail: DisplayRootDetail): string {
  return detail._tag === 'WaitingForModel'
    ? `${detail._tag}:${detail.turnStartedAt}`
    : detail._tag
}

function hiddenWaitingInput(input: DetailInput): DetailInput {
  return { ...input, detail: NO_DETAIL }
}

function initialPresentation(input: DetailInput | null, now: number): DetailInput | null {
  return input?.detail._tag === 'WaitingForModel'
    && now < input.detail.turnStartedAt + WAITING_FOR_MODEL_DELAY_MS
    ? hiddenWaitingInput(input)
    : input
}

class RootDetailPresentationStore {
  private snapshot: DetailSnapshot | null
  private detail: DisplayRootDetail | null
  private readonly listeners = new Set<() => void>()

  constructor(input: DetailInput | null) {
    this.snapshot = input === null
      ? null
      : { chainStartedAt: input.chainStartedAt, key: detailKey(input.detail) }
    this.detail = input?.detail ?? null
  }

  readonly subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }

  readonly getSnapshot = (): DetailSnapshot | null => this.snapshot

  getDetail(): DisplayRootDetail | null {
    return this.detail
  }

  updatePayload(input: DetailInput): void {
    if (
      this.snapshot?.chainStartedAt === input.chainStartedAt
      && this.snapshot.key === detailKey(input.detail)
    ) {
      this.detail = input.detail
    }
  }

  commit(input: DetailInput | null): void {
    const nextSnapshot = input === null
      ? null
      : { chainStartedAt: input.chainStartedAt, key: detailKey(input.detail) }
    const unchanged = this.snapshot?.chainStartedAt === nextSnapshot?.chainStartedAt
      && this.snapshot?.key === nextSnapshot?.key
    this.snapshot = nextSnapshot
    this.detail = input?.detail ?? null
    if (!unchanged) this.listeners.forEach((listener) => listener())
  }
}

/**
 * Stabilizes semantic detail transitions while leaving same-detail payloads
 * live. The initial model wait has a turn-relative one-second threshold.
 */
export function useStabilizedRootDetail(
  status: DisplayRootStatus | null,
  settleMs: number = DEFAULT_SETTLE_MS,
): DisplayRootDetail | null {
  const input: DetailInput | null = status?._tag === 'Working'
    ? { chainStartedAt: status.chainStartedAt, detail: status.detail }
    : null
  const latestInputRef = useRef(input)
  latestInputRef.current = input

  const storeRef = useRef<RootDetailPresentationStore | null>(null)
  if (storeRef.current === null) {
    storeRef.current = new RootDetailPresentationStore(initialPresentation(input, Date.now()))
  }
  const store = storeRef.current
  const chainStartedAt = input?.chainStartedAt ?? null
  const inputKey = input === null ? null : detailKey(input.detail)
  const waitingDeadline = input?.detail._tag === 'WaitingForModel'
    ? input.detail.turnStartedAt + WAITING_FOR_MODEL_DELAY_MS
    : null

  const transitionAtom = useMemo(() => Atom.make(Effect.gen(function* () {
    if (chainStartedAt === null || inputKey === null) {
      store.commit(null)
      return
    }

    const current = store.getSnapshot()
    if (current?.chainStartedAt === chainStartedAt && current.key === inputKey) return

    if (waitingDeadline !== null) {
      const now = yield* Clock.currentTimeMillis
      const remainingMs = waitingDeadline - now
      if (remainingMs > 0) yield* Effect.sleep(remainingMs)
    } else if (current !== null && current.chainStartedAt === chainStartedAt) {
      yield* Effect.sleep(settleMs)
    }

    const latest = latestInputRef.current
    if (
      latest !== null
      && latest.chainStartedAt === chainStartedAt
      && detailKey(latest.detail) === inputKey
    ) {
      store.commit(latest)
    }
  })).pipe(Atom.setIdleTTL(0)), [chainStartedAt, inputKey, settleMs, store, waitingDeadline])
  useAtomMount(transitionAtom)

  const payloadAtom = useMemo(() => Atom.make(Effect.sync(() => {
    const latest = latestInputRef.current
    if (latest !== null) store.updatePayload(latest)
  })).pipe(Atom.setIdleTTL(0)), [input?.detail, store])
  useAtomMount(payloadAtom)

  const committed = useSyncExternalStore(store.subscribe, store.getSnapshot, store.getSnapshot)
  if (input === null) return null
  if (committed === null || committed.chainStartedAt !== input.chainStartedAt) {
    return initialPresentation(input, Date.now())?.detail ?? null
  }
  return committed.key === detailKey(input.detail)
    ? input.detail
    : store.getDetail()
}
