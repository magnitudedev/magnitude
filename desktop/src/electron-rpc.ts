import type { IpcMain, IpcMainEvent, IpcRenderer, IpcRendererEvent, WebContents } from "electron"
import { RpcClient, RpcClientError, RpcServer } from "@effect/rpc"
import type { FromClientEncoded, FromServerEncoded } from "@effect/rpc/RpcMessage"
import { Effect, Layer, Mailbox, Option, Runtime, Schema, Scope } from "effect"
import { randomUUID } from "node:crypto"
import { DesktopRpcChannel, DesktopRpcEnvelope, DesktopRendererSession } from "./desktop-rpc"

const ipcTransportError = (message: string, cause: unknown): RpcClientError.RpcClientError =>
  new RpcClientError.RpcClientError({
    reason: "Protocol",
    message,
    cause,
  })

export const makeElectronRpcClientLayer = (
  ipcRenderer: IpcRenderer,
): Layer.Layer<RpcClient.Protocol, never, never> => {
  const session = DesktopRendererSession.make(randomUUID())
  return Layer.succeed(
    RpcClient.Protocol,
    {
      run: (writeResponse) =>
        Effect.gen(function* () {
          const runtime = yield* Effect.runtime<never>()
          const onResponse = (_event: IpcRendererEvent, response: unknown): void => {
            const envelope = Schema.decodeUnknownEither(DesktopRpcEnvelope)(response)
            if (envelope._tag === "Left" || envelope.right.session !== session) return
            Runtime.runFork(runtime)(writeResponse(envelope.right.message as FromServerEncoded))
          }

          yield* Effect.sync(() => ipcRenderer.on(DesktopRpcChannel.response, onResponse))
          return yield* Effect.onExit(
            Effect.never,
            () => Effect.sync(() => ipcRenderer.removeListener(DesktopRpcChannel.response, onResponse)),
          )
        }),
      send: (request: FromClientEncoded) =>
        Effect.try({
          try: () => ipcRenderer.send(DesktopRpcChannel.request, { session, message: request }),
          catch: (cause) => ipcTransportError("Failed to send desktop RPC request", cause),
        }),
      supportsAck: false,
      supportsTransferables: false,
    },
  )
}

export const makeElectronRpcServerLayer = (
  ipcMain: IpcMain,
): Layer.Layer<RpcServer.Protocol, never, never> =>
  Layer.scoped(
    RpcServer.Protocol,
    RpcServer.Protocol.make((writeRequest) =>
      Effect.gen(function* () {
        const scope = yield* Effect.scope
        const runtime = yield* Effect.runtime<never>()
        const disconnects = yield* Mailbox.make<number>()
        type Client = { readonly id: number; readonly session: typeof DesktopRendererSession.Type; readonly contents: WebContents }
        type WindowClients = { active: Client | undefined; readonly seen: Set<string>; readonly dispose: () => void }
        const clients = new Map<number, Client>()
        const windows = new Map<number, WindowClients>()
        let nextClientId = 0

        const forgetClient = (clientId: number): void => {
          const client = clients.get(clientId)
          if (!client) return
          clients.delete(clientId)
          const window = windows.get(client.contents.id)
          if (window?.active?.id === clientId) window.active = undefined
          Runtime.runFork(runtime)(disconnects.offer(clientId).pipe(Effect.asVoid))
        }

        const trackClient = (webContents: WebContents, session: typeof DesktopRendererSession.Type): number | undefined => {
          let window = windows.get(webContents.id)
          if (!window) {
            const retire = () => { const active = windows.get(webContents.id)?.active; if (active) forgetClient(active.id) }
            const navigation = (details: Electron.Event<Electron.WebContentsDidStartNavigationEventParams>) => {
              if (details.isMainFrame && !details.isSameDocument) retire()
            }
            const destroyed = () => { retire(); windows.get(webContents.id)?.dispose(); windows.delete(webContents.id) }
            const dispose = () => {
              webContents.removeListener("render-process-gone", retire)
              webContents.removeListener("did-start-navigation", navigation)
              webContents.removeListener("destroyed", destroyed)
            }
            window = { active: undefined, seen: new Set(), dispose }
            windows.set(webContents.id, window)
            webContents.on("render-process-gone", retire)
            webContents.on("did-start-navigation", navigation)
            webContents.once("destroyed", destroyed)
          }
          if (window.active?.session === session) return window.active.id
          if (window.seen.has(session)) return undefined
          if (window.active) forgetClient(window.active.id)
          const client = { id: ++nextClientId, session, contents: webContents }
          window.seen.add(session)
          window.active = client
          clients.set(client.id, client)
          return client.id
        }

        const onRequest = (event: IpcMainEvent, request: unknown): void => {
          if (event.senderFrame !== event.sender.mainFrame) return
          const envelope = Schema.decodeUnknownEither(DesktopRpcEnvelope)(request)
          if (envelope._tag === "Left") return
          const clientId = trackClient(event.sender, envelope.right.session)
          if (clientId === undefined) return
          Runtime.runFork(runtime)(
            Effect.suspend(() => clients.has(clientId)
              ? writeRequest(clientId, envelope.right.message as FromClientEncoded)
              : Effect.void).pipe(
              Effect.catchAllCause((cause) =>
                Effect.sync(() => {
                  console.error("[desktop] Desktop RPC request failed:", cause)
                })
              ),
            ),
          )
        }

        yield* Effect.sync(() => ipcMain.on(DesktopRpcChannel.request, onRequest))
        yield* Scope.addFinalizer(
          scope,
          Effect.sync(() => {
            ipcMain.removeListener(DesktopRpcChannel.request, onRequest)
            for (const window of windows.values()) window.dispose()
            windows.clear()
            clients.clear()
          }),
        )

        return {
          disconnects,
          send: (clientId: number, response: FromServerEncoded) =>
            Effect.sync(() => {
              const client = clients.get(clientId)
              if (!client || client.contents.isDestroyed()) return
              client.contents.send(DesktopRpcChannel.response, { session: client.session, message: response })
            }).pipe(Effect.catchAllCause(() => Effect.void)),
          end: (clientId: number) => Effect.sync(() => forgetClient(clientId)),
          clientIds: Effect.sync(() => new Set(clients.keys())),
          initialMessage: Effect.succeed(Option.none()),
          supportsAck: false,
          supportsTransferables: false,
          supportsSpanPropagation: true,
        }
      }),
    ),
  )
