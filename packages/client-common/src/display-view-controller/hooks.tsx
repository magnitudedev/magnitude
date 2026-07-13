import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useRef,
  useSyncExternalStore,
  type ReactNode,
} from "react"
import { Atom, useAtomMount, useAtomSet } from "@effect-atom/atom-react"
import { Effect } from "effect"
import type { DisplayState } from "@magnitudedev/sdk"
import { usePlatform } from "../platform/platform-context"
import {
  createDisplayViewStore,
  EMPTY_DISPLAY_VIEW_SHAPE,
  getFork,
  useDisplayView,
  useDisplaySpeculator,
  type DisplayViewStore,
} from "../sync/index"
import { DisplayReaderContext, DisplaySpeculatorContext } from "../sync/use-display-view"
import { composerTextAtom, composerAttachmentsAtom, composerHistoryIndexAtom } from "../state/session-atoms"
import { EMPTY_DISPLAY_STATE } from "../state/empty-display-state"
import {
  DisplayViewControllerCore,
  desiredShapeForSnapshot,
  sameDisplayShape,
  timelineStatusFor,
  type DisplayConnectionError,
  type DisplayMode,
  type DisplayViewConnectionPhase,
  type DisplayViewControllerSnapshot,
  type TimelineStatus,
} from "./controller"

const DisplayViewControllerContext = createContext<DisplayViewControllerCore | null>(null)

let activeController: DisplayViewControllerCore | null = null

export interface DisplayViewControllerProviderProps {
  readonly children: ReactNode
  readonly initial?: DisplayState
}

export function DisplayViewControllerProvider({
  children,
  initial = EMPTY_DISPLAY_STATE,
}: DisplayViewControllerProviderProps): ReactNode {
  const platform = usePlatform()
  const setComposerText = useAtomSet(composerTextAtom)
  const setComposerAttachments = useAtomSet(composerAttachmentsAtom)
  const setComposerHistoryIndex = useAtomSet(composerHistoryIndexAtom)
  const store = useMemo(
    () => createDisplayViewStore(initial, EMPTY_DISPLAY_VIEW_SHAPE),
    [initial],
  )
  const controller = useMemo(
    () =>
      new DisplayViewControllerCore({
        daemonSpawnerLayer: platform.daemonSpawnerLayer,
        displaySync: store,
        onRestoreQueuedInputText: (text) => {
          if (text != null) {
            setComposerText(text)
            setComposerAttachments([])
            setComposerHistoryIndex(-1)
          } else {
            setComposerText("")
            setComposerAttachments([])
            setComposerHistoryIndex(-1)
          }
        },
      }),
    [platform.daemonSpawnerLayer, setComposerText, setComposerAttachments, setComposerHistoryIndex, store],
  )

  const lifecycleAtom = useMemo(
    () =>
      Atom.make(
        Effect.gen(function* () {
          activeController = controller
          yield* Effect.addFinalizer(() =>
            Effect.sync(() => {
              if (activeController === controller) {
                activeController = null
              }
              controller.dispose()
            }),
          )
        }),
      ),
    [controller],
  )
  useAtomMount(lifecycleAtom)

  return (
    <DisplayViewControllerContext.Provider value={controller}>
      <DisplayReaderContext.Provider value={store}>
        <DisplaySpeculatorContext.Provider value={store}>
          {children}
        </DisplaySpeculatorContext.Provider>
      </DisplayReaderContext.Provider>
    </DisplayViewControllerContext.Provider>
  )
}

export function stopDisplayViewController(): void {
  activeController?.stop()
}

export function useDisplayViewControllerCore(): DisplayViewControllerCore {
  const controller = useContext(DisplayViewControllerContext)
  if (!controller) {
    throw new Error("useDisplayViewController must be used within DisplayViewControllerProvider")
  }
  return controller
}

function useSelectedExternalStore<S, T>(
  subscribe: (listener: () => void) => () => void,
  getSourceSnapshot: () => S,
  selector: (snapshot: S) => T,
  isEqual: (left: T, right: T) => boolean = Object.is,
): T {
  const cacheRef = useRef<{
    readonly source: S
    readonly selector: (snapshot: S) => T
    readonly selected: T
  } | null>(null)

  const getSnapshot = useCallback(() => {
    const source = getSourceSnapshot()
    const cached = cacheRef.current
    if (cached && cached.source === source && cached.selector === selector) {
      return cached.selected
    }

    const selected = selector(source)
    if (cached && isEqual(cached.selected, selected)) {
      cacheRef.current = { source, selector, selected: cached.selected }
      return cached.selected
    }

    cacheRef.current = { source, selector, selected }
    return selected
  }, [getSourceSnapshot, isEqual, selector])

  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot)
}

export function useDisplayControllerSelector<T>(
  selector: (snapshot: DisplayViewControllerSnapshot) => T,
  isEqual?: (left: T, right: T) => boolean,
): T {
  const controller = useDisplayViewControllerCore()
  return useSelectedExternalStore(
    controller.subscribe,
    controller.getSnapshot,
    selector,
    isEqual,
  )
}

export function useDisplayState<T>(
  selector: (state: DisplayState) => T,
  isEqual?: (left: T, right: T) => boolean,
): T {
  return useDisplayView((view) => selector(view.state), isEqual)
}

export interface DisplayViewController {
  readonly selectedSessionId: string | null
  readonly expandedForkStack: readonly string[]
  readonly topForkId: string | null
  readonly displayMode: DisplayMode
  readonly desiredShape: ReturnType<typeof desiredShapeForSnapshot>
  readonly phase: DisplayViewConnectionPhase
  readonly hasReceivedDisplay: boolean
  readonly connectionError: DisplayConnectionError | null
  readonly selectSession: (sessionId: string) => void
  readonly clearSession: () => void
  readonly pushFork: (forkId: string) => void
  readonly popFork: () => void
  readonly setForkStack: (forkIds: readonly string[]) => void
  readonly setPresentationMode: (mode: DisplayMode) => void
  readonly togglePresentationMode: () => void
  readonly retry: () => boolean
  readonly resync: () => void
  readonly stop: () => void
}

export function useDisplayViewController(): DisplayViewController {
  const controller = useDisplayViewControllerCore()
  const speculator = useDisplaySpeculator()
  const snapshot = useDisplayControllerSelector((state) => state)
  const hasReceivedDisplay = useHasReceivedDisplay()
  const desiredShape = useMemo(() => desiredShapeForSnapshot(snapshot), [snapshot])
  const topForkId = snapshot.expandedForkStack[snapshot.expandedForkStack.length - 1] ?? null

  const selectSession = useCallback((sessionId: string) => {
    speculator.clear()
    controller.selectSession(sessionId)
  }, [controller, speculator])

  const clearSession = useCallback(() => {
    speculator.clear()
    controller.clearSession()
  }, [controller, speculator])

  return useMemo(
    () => ({
      selectedSessionId: snapshot.selectedSessionId,
      expandedForkStack: snapshot.expandedForkStack,
      topForkId,
      displayMode: snapshot.displayMode,
      desiredShape,
      phase: snapshot.phase,
      hasReceivedDisplay,
      connectionError: snapshot.connectionError,
      selectSession,
      clearSession,
      pushFork: controller.pushFork,
      popFork: controller.popFork,
      setForkStack: controller.setForkStack,
      setPresentationMode: controller.setPresentationMode,
      togglePresentationMode: controller.togglePresentationMode,
      retry: controller.retry,
      resync: controller.resync,
      stop: controller.stop,
    }),
    [clearSession, controller, desiredShape, hasReceivedDisplay, selectSession, snapshot, topForkId],
  )
}

export function useSelectedSessionId(): string | null {
  return useDisplayControllerSelector((snapshot) => snapshot.selectedSessionId)
}

/** True while a root timeline shape change (history load / evict) is in flight. */
export function useRootHistoryLoading(): boolean {
  const rootTailLimit = useDisplayControllerSelector((snapshot) => snapshot.rootTailLimit)
  const acceptedRootLimit = useDisplayView((view) => {
    const root = view.shape.timelines.root
    return root !== undefined && root.kind === "tail" ? root.limit : null
  })
  return acceptedRootLimit !== null && rootTailLimit !== acceptedRootLimit
}

export function useDisplayConnectionError(): DisplayConnectionError | null {
  return useDisplayControllerSelector((snapshot) => snapshot.connectionError)
}

export function useHasReceivedDisplay(): boolean {
  return useDisplayControllerSelector(
    (snapshot) => snapshot.selectedSessionId === null || snapshot.hasReceivedDisplay,
  )
}

export function useTimelineStatus(forkId: string | null): TimelineStatus {
  const intent = useDisplayControllerSelector(
    (snapshot) => ({
      selectedSessionId: snapshot.selectedSessionId,
      desiredShape: desiredShapeForSnapshot(snapshot),
    }),
    (left, right) =>
      left.selectedSessionId === right.selectedSessionId &&
      sameDisplayShape(left.desiredShape, right.desiredShape),
  )
  const displayView = useDisplayView((view) => view)

  return useMemo(() => {
    const fork = getFork(displayView.state, forkId)
    return timelineStatusFor(
      intent.selectedSessionId,
      intent.desiredShape,
      displayView.shape,
      fork ?? undefined,
      forkId,
    )
  }, [displayView, forkId, intent])
}

export type {
  DisplayConnectionError,
  DisplayMode,
  DisplayViewConnectionPhase,
  DisplayViewControllerSnapshot,
  DisplayViewStore,
  TimelineStatus,
}
