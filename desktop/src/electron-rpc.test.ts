import { EventEmitter } from "node:events"
import { randomUUID } from "node:crypto"
import type { IpcMain, IpcRenderer } from "electron"
import { RpcClient, RpcServer } from "@effect/rpc"
import type { FromClientEncoded, FromServerEncoded } from "@effect/rpc/RpcMessage"
import { Effect, Queue } from "effect"
import { describe, expect, it } from "vitest"
import { makeElectronRpcClientLayer, makeElectronRpcServerLayer } from "./electron-rpc"
import { DesktopRpcChannel } from "./desktop-rpc"

describe("renderer-scoped Electron RPC", () => {
  it("retires subscriptions on crash/navigation and isolates delayed requests and replies", async () => {
    const main = new EventEmitter()
    const frame = {}
    const responses: unknown[] = []
    const contents = Object.assign(new EventEmitter(), {
      id: 7, mainFrame: frame, isDestroyed: () => false,
      send: (_channel: string, response: unknown) => { responses.push(response) },
    })
    const event = { sender: contents, senderFrame: frame }
    const session1 = randomUUID(), session2 = randomUUID(), session3 = randomUUID()
    const send = (session: string, senderFrame = frame) => main.emit(DesktopRpcChannel.request,
      { ...event, senderFrame }, { session, message: { _tag: "Ping" } })
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const protocol = yield* RpcServer.Protocol
      const requests = yield* Queue.unbounded<readonly [number, FromClientEncoded]>()
      yield* protocol.run((id, message) => Queue.offer(requests, [id, message]).pipe(Effect.asVoid)).pipe(Effect.forkScoped)
      send(session1)
      const [first] = yield* Queue.take(requests)
      contents.emit("did-start-navigation", { isMainFrame: true, isSameDocument: true })
      expect([...(yield* protocol.clientIds)]).toEqual([first])
      contents.emit("render-process-gone")
      expect(yield* protocol.disconnects.take).toBe(first)
      expect([...(yield* protocol.clientIds)]).toEqual([])
      send(session2)
      const [second] = yield* Queue.take(requests)
      expect(second).not.toBe(first)
      send(session1)
      send(session3, {})
      yield* Effect.yieldNow()
      expect(yield* Queue.size(requests)).toBe(0)
      yield* protocol.send(first, { _tag: "Pong" })
      expect(responses).toEqual([])
      yield* protocol.send(second, { _tag: "Pong" })
      expect(responses).toEqual([{ session: session2, message: { _tag: "Pong" } }])
      contents.emit("did-start-navigation", { isMainFrame: true, isSameDocument: false })
      expect(yield* protocol.disconnects.take).toBe(second)
      send(session3)
      const [third] = yield* Queue.take(requests)
      expect(third).not.toBe(second)
      contents.emit("destroyed")
      expect(yield* protocol.disconnects.take).toBe(third)
      expect([...(yield* protocol.clientIds)]).toEqual([])
    }).pipe(Effect.provide(makeElectronRpcServerLayer(main as unknown as IpcMain)), Effect.timeout("3 seconds"))))
    expect(main.listenerCount(DesktopRpcChannel.request)).toBe(0)
    expect(contents.listenerCount("render-process-gone")).toBe(0)
    expect(contents.listenerCount("did-start-navigation")).toBe(0)
  })

  it("ignores replies queued for an earlier renderer session", async () => {
    const messages: Array<{ session: string; message: FromClientEncoded }> = []
    const ipc = Object.assign(new EventEmitter(), {
      send: (_channel: string, message: { session: string; message: FromClientEncoded }) => { messages.push(message) },
    })
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const protocol = yield* RpcClient.Protocol
      const responses = yield* Queue.unbounded<FromServerEncoded>()
      yield* protocol.run(message => Queue.offer(responses, message).pipe(Effect.asVoid)).pipe(Effect.forkScoped)
      yield* Effect.yieldNow()
      yield* protocol.send({ _tag: "Ping" })
      ipc.emit(DesktopRpcChannel.response, {}, { session: randomUUID(), message: { _tag: "Pong" } })
      yield* Effect.yieldNow()
      expect(yield* Queue.size(responses)).toBe(0)
      ipc.emit(DesktopRpcChannel.response, {}, { session: messages[0]!.session, message: { _tag: "Pong" } })
      expect(yield* Queue.take(responses)).toEqual({ _tag: "Pong" })
    }).pipe(Effect.provide(makeElectronRpcClientLayer(ipc as unknown as IpcRenderer)), Effect.timeout("3 seconds"))))
    expect(ipc.listenerCount(DesktopRpcChannel.response)).toBe(0)
  })
})
