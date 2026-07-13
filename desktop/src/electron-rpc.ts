import type { IpcMain, IpcMainEvent, IpcRenderer, IpcRendererEvent, WebContents } from "electron"
import { RpcClient, RpcClientError, RpcServer } from "@effect/rpc"
import type { FromClientEncoded, FromServerEncoded } from "@effect/rpc/RpcMessage"
import { Effect, Layer, Mailbox, Option, Runtime, Scope } from "effect"
import { DesktopRpcChannel } from "./desktop-rpc"

const ipcTransportError = (message: string, cause: unknown): RpcClientError.RpcClientError =>
  new RpcClientError.RpcClientError({
    reason: "Protocol",
    message,
    cause,
  })

export const makeElectronRpcClientLayer = (
  ipcRenderer: IpcRenderer,
): Layer.Layer<RpcClient.Protocol, never, never> =>
  Layer.succeed(
    RpcClient.Protocol,
    {
      run: (writeResponse) =>
        Effect.gen(function* () {
          const runtime = yield* Effect.runtime<never>()
          const onResponse = (_event: IpcRendererEvent, response: FromServerEncoded): void => {
            Runtime.runFork(runtime)(writeResponse(response))
          }

          yield* Effect.sync(() => ipcRenderer.on(DesktopRpcChannel.response, onResponse))
          return yield* Effect.onExit(
            Effect.never,
            () => Effect.sync(() => ipcRenderer.removeListener(DesktopRpcChannel.response, onResponse)),
          )
        }),
      send: (request: FromClientEncoded) =>
        Effect.try({
          try: () => ipcRenderer.send(DesktopRpcChannel.request, request),
          catch: (cause) => ipcTransportError("Failed to send desktop RPC request", cause),
        }),
      supportsAck: false,
      supportsTransferables: false,
    },
  )

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
        const clients = new Map<number, WebContents>()
        const clientIds = new Set<number>()

        const forgetClient = (clientId: number): void => {
          clients.delete(clientId)
          clientIds.delete(clientId)
          Runtime.runFork(runtime)(disconnects.offer(clientId).pipe(Effect.asVoid))
        }

        const trackClient = (webContents: WebContents): number => {
          const clientId = webContents.id
          if (!clients.has(clientId)) {
            clients.set(clientId, webContents)
            clientIds.add(clientId)
            webContents.once("destroyed", () => forgetClient(clientId))
          }
          return clientId
        }

        const onRequest = (event: IpcMainEvent, request: FromClientEncoded): void => {
          const clientId = trackClient(event.sender)
          Runtime.runFork(runtime)(
            writeRequest(clientId, request).pipe(
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
          Effect.sync(() => ipcMain.removeListener(DesktopRpcChannel.request, onRequest)),
        )

        return {
          disconnects,
          send: (clientId: number, response: FromServerEncoded) =>
            Effect.sync(() => {
              const webContents = clients.get(clientId)
              if (!webContents || webContents.isDestroyed()) return
              webContents.send(DesktopRpcChannel.response, response)
            }).pipe(Effect.catchAllCause(() => Effect.void)),
          end: (clientId: number) => Effect.sync(() => forgetClient(clientId)),
          clientIds: Effect.sync(() => new Set(clientIds)),
          initialMessage: Effect.succeed(Option.none()),
          supportsAck: false,
          supportsTransferables: false,
          supportsSpanPropagation: true,
        }
      }),
    ),
  )
