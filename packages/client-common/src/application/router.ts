import { Atom, Registry } from "@effect-atom/atom-react"
import { Context, Effect, Layer, Option, Ref, Schema } from "effect"
import { ApplicationPage, type MainPage } from "./contracts"

export interface ApplicationRouter {
  readonly page: Atom.Atom<ApplicationPage>
  /** The page the main area shows; it stays behind Settings while Settings is open over it. */
  readonly mainPage: Atom.Atom<MainPage>
  /**
   * Counts the times a main page was opened from the main area, not returned to by closing Settings over it. A page
   * keyed on it opens at its root, so choosing the page already shown starts it over, as its back arrow would.
   */
  readonly visit: Atom.Atom<number>
  readonly navigate: (page: ApplicationPage) => Effect.Effect<void>
  /** Opens the page the app starts on, unless something has navigated already, such as the tray asking for a page. */
  readonly land: (page: ApplicationPage) => Effect.Effect<void>
}
export const ApplicationRouter = Context.GenericTag<ApplicationRouter>("client/ApplicationRouter")

const DEFAULT_PAGE: ApplicationPage = "chat"
const isPage = Schema.is(ApplicationPage)
const isMainPage = (page: ApplicationPage): page is MainPage => page === "chat" || page === "models" || page === "connections"

export const pagePath = (page: ApplicationPage) => page === DEFAULT_PAGE ? "/" : `/${page}`
export const pageFromPath = (path: string): Option.Option<ApplicationPage> => {
  const segment = path.replace(/^\/+|\/+$/g, "")
  if (segment === "") return Option.some(DEFAULT_PAGE)
  return isPage(segment) ? Option.some(segment) : Option.none()
}

/** The page atoms and the one way they change together. */
const makePages = (registry: Registry.Registry, initial: ApplicationPage) => {
  const page = Atom.keepAlive(Atom.make<ApplicationPage>(initial))
  const mainPage = Atom.keepAlive(Atom.make<MainPage>(isMainPage(initial) ? initial : DEFAULT_PAGE))
  const visit = Atom.keepAlive(Atom.make(0))
  const show = (value: ApplicationPage) => {
    const from = registry.get(page)
    registry.set(page, value)
    if (!isMainPage(value)) return
    registry.set(mainPage, value)
    if (isMainPage(from)) registry.set(visit, registry.get(visit) + 1)
  }
  return { page, mainPage, visit, show }
}

/** In-memory navigation for a native window, where the page is not part of a URL. */
export const ApplicationRouterMemory = Layer.effect(ApplicationRouter, Effect.gen(function* () {
  const registry = yield* Registry.AtomRegistry
  const { page, mainPage, visit, show } = makePages(registry, DEFAULT_PAGE)
  const landable = yield* Ref.make(true)
  return {
    page,
    mainPage,
    visit,
    navigate: value => Ref.set(landable, false).pipe(Effect.zipRight(Effect.sync(() => show(value)))),
    land: value => Effect.whenEffect(Effect.sync(() => show(value)), Ref.getAndSet(landable, false)).pipe(Effect.asVoid),
  }
}))

/**
 * Browser navigation: the page is the URL path, so reload, back, forward and links work. An unknown path is replaced
 * with the default page's. Only a window opened at the root lands on a page of the app's choosing.
 */
export const ApplicationRouterLocation = Layer.scoped(ApplicationRouter, Effect.gen(function* () {
  const registry = yield* Registry.AtomRegistry
  const current = () => Option.getOrElse(pageFromPath(window.location.pathname), () => {
    window.history.replaceState(null, "", pagePath(DEFAULT_PAGE))
    return DEFAULT_PAGE
  })
  const landable = yield* Ref.make(window.location.pathname === "/")
  const { page, mainPage, visit, show } = makePages(registry, current())
  const onPopState = () => {
    Effect.runSync(Ref.set(landable, false))
    show(current())
  }
  yield* Effect.acquireRelease(
    Effect.sync(() => window.addEventListener("popstate", onPopState)),
    () => Effect.sync(() => window.removeEventListener("popstate", onPopState)),
  )
  return {
    page,
    mainPage,
    visit,
    navigate: value => Ref.set(landable, false).pipe(Effect.zipRight(Effect.sync(() => {
      if (window.location.pathname !== pagePath(value)) window.history.pushState(null, "", pagePath(value))
      show(value)
    }))),
    land: value => Effect.whenEffect(Effect.sync(() => {
      window.history.replaceState(null, "", pagePath(value))
      show(value)
    }), Ref.getAndSet(landable, false)).pipe(Effect.asVoid),
  }
}))
