/**
 * Paginated query capability: one authoritative first-page query plus
 * explicitly requested continuation pages, combined into a deduplicated
 * snapshot.
 *
 * Every page — first or later — is a reactive query atom, so reactivity
 * invalidation refreshes the whole retained set; id-dedup absorbs boundary
 * drift between refreshed pages. Queries stay queries: nothing here issues a
 * mutation, and no server truth is copied into component state.
 */
import { Atom, Result } from "@effect-atom/atom-react"
import { Option } from "effect"

export interface PageResult<Item, Cursor> {
  readonly items: ReadonlyArray<Item>
  readonly nextCursor: Option.Option<Cursor>
}

export interface RequestedPage<Cursor> {
  readonly cursor: Cursor
  readonly limit: number | undefined
}

export interface PageSetSnapshot<Item, Cursor> {
  readonly items: ReadonlyArray<Item>
  /** The first page has produced no result yet. */
  readonly loading: boolean
  /** A requested continuation has produced no result yet. */
  readonly loadingMore: boolean
  /** The first page failed with nothing to show. */
  readonly error: boolean
  readonly hasMore: boolean
  readonly nextCursor: Cursor | null
}

export interface PageSet<Item, Cursor> {
  readonly snapshot: Atom.Atom<PageSetSnapshot<Item, Cursor>>
  readonly requestedPages: Atom.Writable<
    ReadonlyArray<RequestedPage<Cursor>>,
    ReadonlyArray<RequestedPage<Cursor>>
  >
}

/**
 * Build one page set. Callers memoize per complete request identity —
 * recreation on an identity change IS the reset.
 */
export const makePageSet = <Item, Cursor>(config: {
  readonly firstPage: Atom.Atom<Result.Result<PageResult<Item, Cursor>, unknown>>
  readonly continuationPage: (
    request: RequestedPage<Cursor>,
  ) => Atom.Atom<Result.Result<PageResult<Item, Cursor>, unknown>>
  readonly itemKey: (item: Item) => string
}): PageSet<Item, Cursor> => {
  const requestedPages = Atom.make<ReadonlyArray<RequestedPage<Cursor>>>([])
  const snapshot = Atom.make((get): PageSetSnapshot<Item, Cursor> => {
    const first = get(config.firstPage)
    const continuations = get(requestedPages).map((request) =>
      get(config.continuationPage(request)))
    const values = [first, ...continuations].map((page) => Result.value(page))

    const items: Item[] = []
    const seen = new Set<string>()
    for (const page of values) {
      if (Option.isNone(page)) continue
      for (const item of page.value.items) {
        const key = config.itemKey(item)
        if (seen.has(key)) continue
        seen.add(key)
        items.push(item)
      }
    }

    const lastLoaded = [...values].reverse().find(Option.isSome)
    const nextCursor = lastLoaded !== undefined && Option.isSome(lastLoaded)
      ? Option.getOrNull(lastLoaded.value.nextCursor)
      : null

    return {
      items,
      loading: Result.isInitial(first),
      loadingMore: continuations.some((page) =>
        !Result.isFailure(page) && Option.isNone(Result.value(page))),
      error: Result.isFailure(first) && Option.isNone(Result.value(first)),
      hasMore: nextCursor !== null,
      nextCursor,
    }
  })
  return { snapshot, requestedPages }
}

/** Idempotent continuation request: one entry per distinct cursor. */
export const appendRequestedPage = <Cursor>(
  current: ReadonlyArray<RequestedPage<Cursor>>,
  cursor: Cursor,
  limit: number | undefined,
): ReadonlyArray<RequestedPage<Cursor>> =>
  current.some((requested) => requested.cursor === cursor)
    ? current
    : [...current, { cursor, limit }]
