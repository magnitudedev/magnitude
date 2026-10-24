import { useMemo } from "react"
import { Atom, useAtomMount, useAtomSet } from "@effect-atom/atom-react"
import { Effect } from "effect"
import { useSession } from "../session"

/** Each Discover view retains its scroll when it is revisited or remounted. */
export function useDiscoverScroll(view: "recommended" | "all", ready: boolean) {
  const session = useSession()
  const node = useMemo(() => Atom.make<HTMLDivElement | null>(null), [])
  const setNode = useAtomSet(node)
  const save = useAtomSet(useMemo(() => Atom.fn((top: number) => session.updateDiscover(view === "all"
    ? { allScroll: top }
    : { recommendedScroll: top })), [session, view]))
  useAtomMount(useMemo(() => Atom.make(get => Effect.gen(function* () {
    const scroller = get(node)?.closest("main")
    if (!ready || !scroller) return
    const state = get.once(session.discover)
    const onScroll = () => save(scroller.scrollTop)
    yield* Effect.acquireRelease(Effect.sync(() => {
      scroller.scrollTop = view === "all" ? state.allScroll : state.recommendedScroll
      scroller.addEventListener("scroll", onScroll, { passive: true })
    }), () => Effect.sync(() => scroller.removeEventListener("scroll", onScroll)))
  })), [node, ready, save, session, view]))
  return setNode
}
