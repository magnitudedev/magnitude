import { FileSystem, HttpServerRequest, HttpServerResponse } from "@effect/platform"
import { Effect, Option } from "effect"
import { extname, join, normalize, resolve, sep } from "node:path"
import { fileURLToPath } from "node:url"
import { embeddedWebApp } from "./web-app-embed"

/** Where the browser app's files come from: embedded in the executable, a build directory, or nowhere. */
export type WebAppSource =
  | { readonly _tag: "Embedded"; readonly files: ReadonlyMap<string, string> }
  | { readonly _tag: "Directory"; readonly root: string }
  | { readonly _tag: "Missing" }

const sourceDirectory = fileURLToPath(new URL("../../../web/dist", import.meta.url))

export const resolveWebAppSource: Effect.Effect<WebAppSource, never, FileSystem.FileSystem> = Effect.gen(function* () {
  if (embeddedWebApp.length > 0) return { _tag: "Embedded", files: new Map(embeddedWebApp.map(entry => [entry.path, entry.file])) } satisfies WebAppSource
  const fs = yield* FileSystem.FileSystem
  const root = process.env.MAGNITUDE_WEB_DIST ?? sourceDirectory
  const present = yield* fs.exists(join(root, "index.html")).pipe(Effect.orElseSucceed(() => false))
  return (present ? { _tag: "Directory", root } : { _tag: "Missing" }) satisfies WebAppSource
})

const CONTENT_TYPES: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".webp": "image/webp",
  ".jpg": "image/jpeg",
  ".woff2": "font/woff2",
  ".woff": "font/woff",
  ".json": "application/json",
  ".ico": "image/x-icon",
}

/** Only same-origin resources; the app talks to this service and nothing else. */
export const WEB_APP_CSP = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"

const NOT_BUILT = [
  "Magnitude local service.",
  "",
  "The browser app has not been built in this checkout. Run `bun run --filter @magnitudedev/web build`,",
  "or start the Vite entry with `bun run --filter @magnitudedev/web dev`.",
  "",
].join("\n")

/** The service's own routes; an unmatched path under them is never an app page. */
const SERVICE_PATHS = ["/inference", "/rpc", "/health", "/auth"]

/** A URL path as a relative file path, or nothing when it could escape the app's files. */
export const webAppFilePath = (pathname: string): Option.Option<string> => {
  let decoded: string
  try {
    decoded = decodeURIComponent(pathname)
  } catch {
    return Option.none()
  }
  if (decoded.includes("\0") || decoded.includes("\\")) return Option.none()
  const relative = normalize(decoded).replace(/^\/+/, "")
  if (relative.split(/[\\/]/).includes("..")) return Option.none()
  return Option.some(relative === "" || relative === "." ? "index.html" : relative)
}

export const serveWebApp = (source: WebAppSource, fs: FileSystem.FileSystem): Effect.Effect<HttpServerResponse.HttpServerResponse, never, HttpServerRequest.HttpServerRequest> => Effect.gen(function* () {
  const request = yield* HttpServerRequest.HttpServerRequest
  if (source._tag === "Missing") return HttpServerResponse.text(NOT_BUILT, { status: 503 })
  const pathname = new URL(request.url, "http://localhost").pathname
  if (SERVICE_PATHS.some(prefix => pathname === prefix || pathname.startsWith(`${prefix}/`))) return HttpServerResponse.text("Not found", { status: 404 })
  const relative = webAppFilePath(pathname)
  if (Option.isNone(relative)) return HttpServerResponse.text("Not found", { status: 404 })
  const locate = (path: string): Option.Option<string> => source._tag === "Embedded"
    ? Option.fromNullable(source.files.get(path))
    : Option.some(join(source.root, path))
  const read = (path: string) => Option.match(locate(path), {
    onNone: () => Effect.succeed(Option.none<Uint8Array>()),
    onSome: file => {
      if (source._tag === "Directory" && !resolve(file).startsWith(resolve(source.root) + sep) && resolve(file) !== resolve(source.root, "index.html")) return Effect.succeed(Option.none<Uint8Array>())
      return fs.readFile(file).pipe(Effect.map(Option.some), Effect.orElseSucceed(() => Option.none<Uint8Array>()))
    },
  })
  const exact = yield* read(relative.value)
  // A path without a file extension is a page of the app; the app routes it.
  const page = Option.isNone(exact) && extname(relative.value) === "" ? yield* read("index.html") : Option.none<Uint8Array>()
  const body = Option.orElse(exact, () => page)
  if (Option.isNone(body)) return HttpServerResponse.text("Not found", { status: 404 })
  const served = Option.isSome(exact) ? relative.value : "index.html"
  const html = served.endsWith(".html")
  return HttpServerResponse.uint8Array(body.value, {
    contentType: CONTENT_TYPES[extname(served)] ?? "application/octet-stream",
    headers: {
      "cache-control": served.startsWith("assets/") ? "public, max-age=31536000, immutable" : "no-cache",
      "x-content-type-options": "nosniff",
      ...(html ? { "content-security-policy": WEB_APP_CSP, "referrer-policy": "no-referrer" } : {}),
    },
  })
})
