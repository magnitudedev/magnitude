import { Atom, Registry } from "@effect-atom/atom-react"
import { Context, Effect, Layer, Option, Schema } from "effect"
import { ApplicationPage } from "./contracts"

export interface ApplicationRouter {
  readonly page: Atom.Atom<ApplicationPage>
  readonly navigate: (page: ApplicationPage) => Effect.Effect<void>
}
export const ApplicationRouter = Context.GenericTag<ApplicationRouter>("client/ApplicationRouter")

const DEFAULT_PAGE: ApplicationPage = "discover"
const isPage = Schema.is(ApplicationPage)

export const pagePath = (page: ApplicationPage) => page === DEFAULT_PAGE ? "/" : `/${page}`
export const pageFromPath = (path: string): Option.Option<ApplicationPage> => {
  const segment = path.replace(/^\/+|\/+$/g, "")
  if (segment === "") return Option.some(DEFAULT_PAGE)
  return isPage(segment) ? Option.some(segment) : Option.none()
}

/** In-memory navigation for a native window, where the page is not part of a URL. */
export const ApplicationRouterMemory = Layer.effect(ApplicationRouter, Effect.gen(function* () {
  const registry = yield* Registry.AtomRegistry
  const page = Atom.keepAlive(Atom.make<ApplicationPage>(DEFAULT_PAGE))
  return { page, navigate: value => Effect.sync(() => registry.set(page, value)) }
}))

/** Browser navigation: the page is the URL path, so reload, back, forward and links work. */
export const ApplicationRouterLocation = Layer.scoped(ApplicationRouter, Effect.gen(function* () {
  const registry = yield* Registry.AtomRegistry
  const current = () => Option.getOrElse(pageFromPath(window.location.pathname), () => DEFAULT_PAGE)
  const page = Atom.keepAlive(Atom.make<ApplicationPage>(current()))
  const onPopState = () => registry.set(page, current())
  yield* Effect.acquireRelease(
    Effect.sync(() => window.addEventListener("popstate", onPopState)),
    () => Effect.sync(() => window.removeEventListener("popstate", onPopState)),
  )
  return {
    page,
    navigate: value => Effect.sync(() => {
      if (window.location.pathname !== pagePath(value)) window.history.pushState(null, "", pagePath(value))
      registry.set(page, value)
    }),
  }
}))
