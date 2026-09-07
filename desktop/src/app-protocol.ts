import * as nodePath from "node:path"
import { MAGNITUDE_APP_ORIGIN } from "@magnitudedev/sdk"

export const MAGNITUDE_APP_SCHEME = new URL(MAGNITUDE_APP_ORIGIN).protocol.slice(0, -1)

export const resolveMagnitudeAppAssetPath = (
  rendererRoot: string,
  requestUrl: string,
): string | null => {
  let url: URL
  let pathname: string
  try {
    url = new URL(requestUrl)
    pathname = decodeURIComponent(url.pathname)
  } catch {
    return null
  }

  if (url.protocol !== `${MAGNITUDE_APP_SCHEME}:` || url.host !== "app") return null

  const requestedPath = pathname === "/" ? "/index.html" : pathname
  const assetPath = nodePath.resolve(rendererRoot, `.${requestedPath}`)
  const relativePath = nodePath.relative(rendererRoot, assetPath)
  if (relativePath.startsWith("..") || nodePath.isAbsolute(relativePath)) return null
  return assetPath
}
