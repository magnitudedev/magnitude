/**
 * Preload bridge — spec §5.4
 *
 * Exposes a typed DesktopApi to the renderer via contextBridge.
 * Effect RPC stays inside the preload layer; the renderer receives the same
 * narrow desktop facade for platform actions and daemon lifecycle.
 */
import { contextBridge, ipcRenderer, clipboard as electronClipboard, shell } from "electron"
import { RpcClient } from "@effect/rpc"
import { Context, Effect, Fiber, Layer, ManagedRuntime, Stream } from "effect"
import {
  DesktopRpcError,
  DesktopRpcs,
  type DesktopApi,
  type DesktopPlatform,
  type DesktopRpcClient,
  type MenuAction,
} from "./desktop-rpc"
import { makeElectronRpcClientLayer } from "./electron-rpc"

function errorMessage(cause: unknown): string {
  if (cause instanceof DesktopRpcError) return cause.message
  if (cause instanceof Error) return cause.message
  if (typeof cause === "object" && cause !== null && "message" in cause) {
    return String(cause.message)
  }
  return String(cause)
}

class DesktopRpcClientTag extends Context.Tag("DesktopRpcClient")<
  DesktopRpcClientTag,
  DesktopRpcClient
>() {}

function makeDesktopRpcRuntime() {
  const runtime = ManagedRuntime.make(
    Layer.scoped(DesktopRpcClientTag, RpcClient.make(DesktopRpcs)).pipe(
      Layer.provide(makeElectronRpcClientLayer(ipcRenderer)),
    ),
  )
  const clientPromise = runtime.runPromise(DesktopRpcClientTag)

  return {
    async run<A>(
      operation: (client: DesktopRpcClient) => Effect.Effect<A, unknown, never>,
    ): Promise<A> {
      const client = await clientPromise
      try {
        return await runtime.runPromise(operation(client))
      } catch (cause) {
        throw new Error(errorMessage(cause))
      }
    },
    onMenuAction(cb: (action: MenuAction) => void): () => void {
      let active = true
      let fiber: Fiber.RuntimeFiber<void, unknown> | null = null

      void clientPromise.then((client) => {
        if (!active) return
        fiber = runtime.runFork(
          client.StreamMenuActions({}).pipe(
            Stream.runForEach((action) => Effect.sync(() => cb(action))),
            Effect.catchAllCause((cause) =>
              Effect.sync(() => {
                console.error("[desktop] Menu action stream failed:", cause)
              })
            ),
          ),
        )
      }).catch((cause) => {
        console.error("[desktop] Failed to start menu action stream:", cause)
      })

      return () => {
        active = false
        if (fiber) {
          runtime.runFork(Fiber.interrupt(fiber))
          fiber = null
        }
      }
    },
  }
}

function makeDesktopApi(): DesktopApi {
  let daemonErrorResolve: ((message: string) => void) | null = null
  const daemonErrorPromise = new Promise<string>((resolve) => {
    daemonErrorResolve = resolve
  })
  const desktopRpc = makeDesktopRpcRuntime()
  const readyPromise = desktopRpc.run((client) => client.DaemonEnsure({}))
    .then(() => undefined)
    .catch((cause: unknown) => {
      const message = errorMessage(cause)
      daemonErrorResolve?.(message)
      daemonErrorResolve = null
      throw new Error(message)
    })

  return {
    get ready(): Promise<void> {
      return readyPromise
    },
    get daemonError(): Promise<string> {
      return daemonErrorPromise
    },
    get platform(): DesktopPlatform {
      return process.platform as DesktopPlatform
    },
    daemon: {
      async discover(): Promise<string | null> {
        return desktopRpc.run((client) => client.DaemonDiscover({}))
      },
      async spawn(command?: string[]): Promise<string> {
        return desktopRpc.run((client) => client.DaemonSpawn({ command: command ?? null }))
      },
    },
    onMenuAction(cb: (action: MenuAction) => void): () => void {
      return desktopRpc.onMenuAction(cb)
    },
    quit(): void {
      void desktopRpc.run((client) => client.Quit({})).catch((cause) => {
        console.error("[desktop] Quit RPC failed:", cause)
      })
    },
    interruptStream(): void {
      void desktopRpc.run((client) => client.InterruptStream({})).catch((cause) => {
        console.error("[desktop] Interrupt stream RPC failed:", cause)
      })
    },
    async openPath(path: string): Promise<void> {
      await shell.openPath(path)
    },
    async openExternal(url: string): Promise<void> {
      await shell.openExternal(url)
    },
    showItemInFolder(path: string): void {
      shell.showItemInFolder(path)
    },
    storage: {
      async getItem(key: string): Promise<string | null> {
        return desktopRpc.run((client) => client.StorageGet({ key }))
      },
      async setItem(key: string, value: string): Promise<void> {
        await desktopRpc.run((client) => client.StorageSet({ key, value }))
      },
      async removeItem(key: string): Promise<void> {
        await desktopRpc.run((client) => client.StorageRemove({ key }))
      },
    },
    clipboard: {
      async readText(): Promise<string> {
        return electronClipboard.readText()
      },
      async writeText(text: string): Promise<void> {
        electronClipboard.writeText(text)
      },
    },
    dialogs: {
      async openDirectory(): Promise<string | null> {
        return desktopRpc.run((client) => client.DialogOpenDirectory({}))
      },
      async openFile(options?: { multiple?: boolean }): Promise<string[] | null> {
        const paths = await desktopRpc.run((client) => client.DialogOpenFile({ multiple: options?.multiple ?? false }))
        return paths === null ? null : [...paths]
      },
    },
    notifications: {
      show(title: string, body: string): void {
        void desktopRpc.run((client) => client.NotificationShow({ title, body })).catch((cause) => {
          console.error("[desktop] Notification RPC failed:", cause)
        })
      },
    },
  }
}

contextBridge.exposeInMainWorld("__magnitudeDesktop", makeDesktopApi())
