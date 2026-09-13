import { ApplicationUpdateControlFailed, DesktopUpdateState, LoginStartupFailed, type LoginStartupState } from "@magnitudedev/sdk/desktop-host"
import { mkdtemp, rm, writeFile, stat } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { Effect, Fiber, Ref, Schema } from "effect"
import { describe, expect, it, vi } from "vitest"
import { createServer as createHttpServer } from "node:http"
import { createServer as createLocalServer, Socket } from "node:net"
import { ApplicationSnapshot, requestLoginStartup, requestApplicationUpdate, requestApplication, serveApplicationControl, type ApplicationIntent } from "./application-control"

const snapshot = Schema.decodeUnknownSync(ApplicationSnapshot)({ version: 1, pid: process.pid, endpoint: "http://127.0.0.1:11101", service: { _tag: "Starting", attempt: 0 }, tray: { _tag: "Registered" } })
const setup = Effect.acquireRelease(Effect.promise(() => mkdtemp(join(tmpdir(), "mag-ipc-"))), path => Effect.promise(() => rm(path, { recursive: true, force: true })))
describe("local application control", () => {
  it("carries update state independently of service readiness and acknowledges before restart", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const root = yield* setup
      const path = join(root, "application.sock")
      const ready = DesktopUpdateState.make({ transfer: { _tag: "Ready", version: "2.0.0" }, check: { _tag: "Succeeded", at: 1 }, preference: { _tag: "Known", autoDownload: false } })
      const actions = yield* Ref.make<string[]>([])
      yield* serveApplicationControl(path, { snapshot: Effect.succeed(snapshot), login: () => Effect.die("Unexpected login request"), dispatch: () => Effect.die("Updates cannot show a window"),
        update: action => action === "download" ? new ApplicationUpdateControlFailed({ message: "Already prepared" }) : Effect.succeed({ state: ready, afterReply: Ref.update(actions, values => [...values, action]) }),
      })
      expect(yield* requestApplicationUpdate(path, "status")).toEqual(ready)
      expect(yield* requestApplicationUpdate(path, "install")).toEqual(ready)
      const refused = yield* requestApplicationUpdate(path, "download").pipe(Effect.either)
      expect(refused._tag).toBe("Left")
      if (refused._tag === "Left") expect(refused.left.message).toBe("Already prepared")
      expect(yield* Ref.get(actions)).toEqual(["status", "install"])
    })))
  })
  it.each([`/tmp/${"模型".repeat(30)}.sock`, "/tmp/magnitude\0ignored.sock"])("rejects invalid client path %j before connecting for every request", async path => {
    const connect = vi.spyOn(Socket.prototype, "connect")
    try {
      const requests = [
        ...(["EnsureRunning", "ShowWindow", "Observe", "Retry", "Quit"] as const).map(intent => requestApplication(path, intent).pipe(Effect.asVoid)),
        ...(["read", "enable", "disable"] as const).map(action => requestLoginStartup(path, action).pipe(Effect.asVoid)),
      ]
      for (const request of requests) {
        const result = await Effect.runPromise(Effect.either(request))
        expect(result._tag).toBe("Left")
        if (result._tag === "Left") {
          expect(result.left._tag).toBe("ApplicationControlFailed")
          expect(result.left.message).toContain("Use a shorter application state directory")
        }
      }
      expect(connect).not.toHaveBeenCalled()
    } finally {
      connect.mockRestore()
    }
  })
  it("rejects overlong UTF-8 socket paths before attempting filesystem changes", async () => {
    const result = await Effect.runPromise(Effect.scoped(Effect.either(serveApplicationControl(`/tmp/${"模型".repeat(30)}.sock`, {
      snapshot: Effect.succeed(snapshot), update: () => Effect.die("Unexpected update request"), login: () => Effect.succeed({ _tag: "Disabled" }), dispatch: () => Effect.void,
    }))))
    expect(result._tag).toBe("Left")
    if (result._tag === "Left") expect(result.left.message).toContain("UTF-8 bytes")
  })
  it("reports malformed owner replies as control failure rather than absence", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const path = join(yield* setup, "invalid.sock")
      const server = yield* Effect.acquireRelease(Effect.sync(() => createLocalServer(socket => {
        socket.on("error", () => {})
        socket.once("data", () => socket.write('{"wrong":"contract"}\n'))
      })), server => Effect.promise(() => new Promise<void>(resolve => server.close(() => resolve()))))
      yield* Effect.promise(() => new Promise<void>(resolve => server.listen(path, resolve)))
      const result = yield* Effect.either(requestApplication(path, "Observe"))
      expect(result._tag).toBe("Left")
      if (result._tag === "Left") {
        expect(result.left._tag).toBe("ApplicationControlFailed")
        expect(result.left.message).toBe("Invalid control frame")
      }
    })))
  })
  it("reports cold-start connection failure inside an HTTP callback without crashing the host", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const path = join(yield* setup, "missing.sock")
      const server = yield* Effect.acquireRelease(Effect.sync(() => createHttpServer((_request, response) => {
        void Effect.runPromise(requestApplication(path, "EnsureRunning").pipe(Effect.either)).then(result => {
          response.end(result._tag === "Left" ? result.left._tag : "unexpected owner")
        })
      })), server => Effect.promise(() => new Promise<void>((resolve, reject) => server.close(error => error ? reject(error) : resolve()))))
      yield* Effect.promise(() => new Promise<void>(resolve => server.listen(0, "127.0.0.1", resolve)))
      const address = server.address()
      if (address === null || typeof address === "string") return yield* Effect.die("Expected TCP address")
      const result = yield* Effect.promise(() => fetch(`http://127.0.0.1:${address.port}`).then(response => response.text()))
      expect(result).toBe("ApplicationControlUnavailable")
    })))
  })
  it("serves every intent without confusing owner presence with service readiness", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const path = join(yield* setup, "app.sock")
      const intents = yield* Ref.make<ApplicationIntent[]>([])
      const server = yield* serveApplicationControl(path, { snapshot: Effect.succeed(snapshot), update: () => Effect.die("Unexpected update request"), login: () => Effect.succeed({ _tag: "Disabled" }), dispatch: intent => Ref.update(intents, list => [...list, intent]) })
      expect((yield* Effect.promise(() => stat(path))).mode & 0o777).toBe(0o600)
      for (const intent of ["EnsureRunning", "ShowWindow", "Observe", "Retry", "Quit"] as const) {
        const received = yield* requestApplication(path, intent)
        expect(received.service._tag).toBe("Starting")
        expect(received.pid).toBe(process.pid)
      }
      expect(yield* Ref.get(intents)).toEqual(["EnsureRunning", "ShowWindow", "Observe", "Retry", "Quit"])
      yield* Fiber.interrupt(server)
      expect(yield* Effect.promise(() => stat(path).then(() => true, () => false))).toBe(false)
    })))
  })
  it("does not remove ordinary files at the socket path", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const path = join(yield* setup, "app.sock")
      yield* Effect.promise(() => writeFile(path, "keep"))
      const result = yield* serveApplicationControl(path, { snapshot: Effect.succeed(snapshot), update: () => Effect.die("Unexpected update request"), login: () => Effect.succeed({ _tag: "Disabled" }), dispatch: () => Effect.void }).pipe(Effect.either)
      expect(result._tag).toBe("Left")
      expect((yield* Effect.promise(() => stat(path))).isFile()).toBe(true)
    })))
  })
  it("reports absent owners without starting or adopting a service", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const result = yield* requestApplication(join(yield* setup, "missing.sock"), "EnsureRunning").pipe(Effect.either)
      expect(result._tag).toBe("Left")
    })))
  })
  it("acknowledges login changes after the OS adapter completes and propagates errors", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const path = join(yield* setup, "app.sock")
      const state = yield* Ref.make<LoginStartupState>({ _tag: "Disabled" })
      yield* serveApplicationControl(path, { snapshot: Effect.succeed(snapshot), update: () => Effect.die("Unexpected update request"), dispatch: () => Effect.die("Login requests must not dispatch lifecycle intent"), login: action => action === "read" ? Ref.get(state) : action === "enable" ? Ref.set(state, { _tag: "Enabled" }).pipe(Effect.zipRight(Ref.get(state))) : Effect.fail(new LoginStartupFailed({ message: "OS refused change" })) })
      expect((yield* requestLoginStartup(path, "read"))._tag).toBe("Disabled")
      expect((yield* requestLoginStartup(path, "enable"))._tag).toBe("Enabled")
      expect((yield* requestLoginStartup(path, "read"))._tag).toBe("Enabled")
      const result = yield* requestLoginStartup(path, "disable").pipe(Effect.either)
      expect(result._tag).toBe("Left")
      if (result._tag === "Left") expect(result.left.message).toBe("OS refused change")
    })))
  })

})
