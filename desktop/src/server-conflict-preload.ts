import { contextBridge, ipcRenderer } from "electron"

export interface ServerConflictApi {
  readonly platform: NodeJS.Platform
  readonly keep: () => Promise<{ readonly ok: true }>
  readonly stop: () => Promise<{ readonly ok: true } | { readonly ok: false; readonly message: string }>
}

const api: ServerConflictApi = {
  platform: process.platform,
  keep: () => ipcRenderer.invoke("server-conflict:keep"),
  stop: () => ipcRenderer.invoke("server-conflict:stop"),
}
contextBridge.exposeInMainWorld("__magnitudeServerConflict", api)
