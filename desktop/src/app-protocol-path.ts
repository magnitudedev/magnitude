import { join, normalize, sep } from "node:path"

export const DESKTOP_APP_ORIGIN = "magnitude://app"
export const DESKTOP_APP_SCHEME = "magnitude"
export const DESKTOP_APP_HOST = "app"

export const resolveRendererDir = (mainDir: string): string =>
  join(mainDir, "../renderer")

export const resolveAppProtocolPath = (
  url: string,
  rendererDir: string,
): string | null => {
  let parsed: URL
  try {
    parsed = new URL(url)
  } catch {
    return null
  }
  if (parsed.protocol !== `${DESKTOP_APP_SCHEME}:` || parsed.host !== DESKTOP_APP_HOST) {
    return null
  }
  const pathname = decodeURIComponent(parsed.pathname)
  const relative = pathname === "/" ? "/index.html" : pathname
  const absolute = normalize(join(rendererDir, `.${relative}`))
  if (absolute !== rendererDir && !absolute.startsWith(rendererDir + sep)) return null
  return absolute
}

export const mimeForAppProtocolPath = (path: string): string => {
  if (path.endsWith(".html")) return "text/html; charset=utf-8"
  if (path.endsWith(".js") || path.endsWith(".mjs")) return "text/javascript; charset=utf-8"
  if (path.endsWith(".css")) return "text/css; charset=utf-8"
  if (path.endsWith(".json")) return "application/json; charset=utf-8"
  if (path.endsWith(".svg")) return "image/svg+xml"
  if (path.endsWith(".png")) return "image/png"
  if (path.endsWith(".ico")) return "image/x-icon"
  if (path.endsWith(".woff2")) return "font/woff2"
  if (path.endsWith(".woff")) return "font/woff"
  return "application/octet-stream"
}
