import { BunContext, BunHttpServer } from "@effect/platform-bun"
import { FetchHttpClient, FileSystem, HttpClient, HttpServer } from "@effect/platform"
import * as HttpLayerRouter from "@effect/platform/HttpLayerRouter"
import { Context, Effect, Layer, Option } from "effect"
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { describe, expect, it } from "vitest"
import { WEB_APP_CSP, serveWebApp, webAppFilePath, type WebAppSource } from "./web-app"

const build = () => {
  const root = mkdtempSync(join(tmpdir(), "magnitude-web-app-"))
  mkdirSync(join(root, "assets"))
  writeFileSync(join(root, "index.html"), "<!doctype html><div id=root></div>")
  writeFileSync(join(root, "assets", "index-abc.js"), "console.log(1)")
  writeFileSync(join(root, "favicon.svg"), "<svg/>")
  return root
}

const withServer = <A>(source: WebAppSource, use: (origin: string) => Effect.Effect<A, unknown, HttpClient.HttpClient>) =>
  Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const router = yield* HttpLayerRouter.make
    yield* router.add("GET", "/", serveWebApp(source, fs))
    yield* router.add("GET", "/*", serveWebApp(source, fs))
    const infrastructure = yield* Layer.build(BunHttpServer.layer({ hostname: "127.0.0.1", port: 0, idleTimeout: 0 }))
    const server = Context.get(infrastructure, HttpServer.HttpServer)
    yield* server.serve(router.asHttpEffect()).pipe(Effect.provide(infrastructure))
    if (server.address._tag !== "TcpAddress") return yield* Effect.dieMessage("Expected TCP")
    return yield* use(`http://127.0.0.1:${server.address.port}`)
  })).pipe(Effect.provide(Layer.merge(BunContext.layer, FetchHttpClient.layer))))

describe("browser app paths", () => {
  it("maps the root to the app and keeps nested paths", () => {
    expect(webAppFilePath("/")).toEqual(Option.some("index.html"))
    expect(webAppFilePath("/assets/index-abc.js")).toEqual(Option.some("assets/index-abc.js"))
  })
  it("refuses anything that could leave the app's files", () => {
    for (const path of ["/../secret", "/assets/../../secret", "/%2e%2e/secret", "/%2e%2e%2f%2e%2e/secret", "/a%00b", "/a\\b", "/%E0%A4%A"]) {
      const mapped = webAppFilePath(path)
      expect(Option.isNone(mapped) || (!mapped.value.startsWith("/") && !mapped.value.split("/").includes("..")), path).toBe(true)
    }
    expect(webAppFilePath("/a%00b")).toEqual(Option.none())
  })
})

describe("serving the browser app", () => {
  it("serves the app with its security policy and no caching", () => withServer({ _tag: "Directory", root: build() }, origin => Effect.gen(function* () {
    const response = yield* HttpClient.get(`${origin}/`)
    expect(response.status).toBe(200)
    expect(response.headers["content-type"]).toContain("text/html")
    expect(response.headers["content-security-policy"]).toBe(WEB_APP_CSP)
    expect(response.headers["cache-control"]).toBe("no-cache")
  })))
  it("caches hashed assets for good", () => withServer({ _tag: "Directory", root: build() }, origin => Effect.gen(function* () {
    const response = yield* HttpClient.get(`${origin}/assets/index-abc.js`)
    expect(response.status).toBe(200)
    expect(response.headers["content-type"]).toContain("text/javascript")
    expect(response.headers["cache-control"]).toContain("immutable")
    expect(yield* response.text).toBe("console.log(1)")
  })))
  it("lets the app route page paths, and 404s missing files", () => withServer({ _tag: "Directory", root: build() }, origin => Effect.gen(function* () {
    expect(yield* (yield* HttpClient.get(`${origin}/catalog`)).text).toContain("id=root")
    expect((yield* HttpClient.get(`${origin}/assets/missing.js`)).status).toBe(404)
  })))
  it("never answers the service's own paths with the app", () => withServer({ _tag: "Directory", root: build() }, origin => Effect.gen(function* () {
    expect((yield* HttpClient.get(`${origin}/inference/api/v1/models`)).status).toBe(404)
    expect((yield* HttpClient.get(`${origin}/health/extra`)).status).toBe(404)
  })))
  it("explains a checkout without a build", () => withServer({ _tag: "Missing" }, origin => Effect.gen(function* () {
    const response = yield* HttpClient.get(`${origin}/`)
    expect(response.status).toBe(503)
    expect(yield* response.text).toContain("browser app has not been built")
  })))
})
