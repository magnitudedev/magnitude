import { net, protocol } from "electron"
import { pathToFileURL } from "node:url"
import {
  DESKTOP_APP_SCHEME,
  resolveAppProtocolPath,
  mimeForAppProtocolPath,
} from "./app-protocol-path"

export { DESKTOP_APP_ORIGIN, resolveRendererDir } from "./app-protocol-path"

protocol.registerSchemesAsPrivileged([
  {
    scheme: DESKTOP_APP_SCHEME,
    privileges: { standard: true, secure: true, supportFetchAPI: true, corsEnabled: true },
  },
])

export const handleAppProtocol = (rendererDir: string) => {
  protocol.handle(DESKTOP_APP_SCHEME, async (request) => {
    const path = resolveAppProtocolPath(request.url, rendererDir)
    if (path === null) return new Response("Not found", { status: 404 })
    try {
      const fileResponse = await net.fetch(pathToFileURL(path).href)
      if (!fileResponse.ok && path.endsWith(".html")) {
        return new Response("Not found", { status: 404 })
      }
      const headers = new Headers(fileResponse.headers)
      headers.set("content-type", mimeForAppProtocolPath(path))
      headers.set("content-security-policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' data:; connect-src 'self' http://127.0.0.1:* ws://127.0.0.1:* http://localhost:* ws://localhost:*")
      return new Response(await fileResponse.arrayBuffer(), { status: fileResponse.status, headers })
    } catch {
      return new Response("Not found", { status: 404 })
    }
  })
}
