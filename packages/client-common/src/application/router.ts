import { Atom, Registry } from "@effect-atom/atom-react"
import { Context, Effect, Layer, Option, Ref, Schema } from "effect"
import { ApplicationPage, type MainPage } from "./contracts"

export interface ApplicationRouter {
  readonly page: Atom.Atom<ApplicationPage>
  /** Chat or Models, whichever the main area shows; it stays behind Settings while Settings is open over it. */
  readonly mainPage: Atom.Atom<MainPage>
  readonly navigate: (page: ApplicationPage) => Effect.Effect<void>
  /** Opens the page the app starts on, unless something has navigated already or the window opened on a deep link. */
  readonly land: (page: ApplicationPage) => Effect.Effect<void>
}
export const ApplicationRouter = Context.GenericTag<ApplicationRouter>("client/ApplicationRouter")

const DEFAULT_PAGE: ApplicationPage = "chat"
const isPage = Schema.is(ApplicationPage)
const isMainPage = (page: ApplicationPage): page is MainPage => page === "chat" || page === "models"

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
  const show = (value: ApplicationPage) => {
    registry.set(page, value)
    if (isMainPage(value)) registry.set(mainPage, value)
  }
  return { page, mainPage, show }
}

/** In-memory navigation for a native window, where the page is not part of a URL. */
export const ApplicationRouterMemory = Layer.effect(ApplicationRouter, Effect.gen(function* () {
  const registry = yield* Registry.AtomRegistry
  const { page, mainPage, show } = makePages(registry, DEFAULT_PAGE)
  const landable = yield* Ref.make(true)
  return {
    page,
    mainPage,
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
  const { page, mainPage, show } = makePages(registry, current())
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
