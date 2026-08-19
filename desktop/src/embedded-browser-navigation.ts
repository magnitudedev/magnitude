export type BrowserNavigationTarget =
  | { readonly _tag: "navigate"; readonly url: string }
  | { readonly _tag: "insecure"; readonly url: string }
  | { readonly _tag: "invalid"; readonly reason: string }

const SEARCH_ENDPOINT = "https://www.google.com/search?q="
const SCHEME = /^[a-z][a-z\d+.-]*:/i
const HOST_LIKE = /^(?:localhost|\[[0-9a-f:]+\]|(?:\d{1,3}\.){3}\d{1,3}|[^\s/]+\.[^\s/]+)(?::\d+)?(?:[/?#].*)?$/i

export const isLoopbackHostname = (hostname: string): boolean => {
  const normalized = hostname.toLowerCase().replace(/^\[|\]$/g, "")
  if (normalized === "localhost" || normalized.endsWith(".localhost") || normalized === "::1") return true
  const match = /^127\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/.exec(normalized)
  return match !== null && match.slice(1).every((part) => Number(part) <= 255)
}

const classifyUrl = (candidate: string): BrowserNavigationTarget => {
  let url: URL
  try {
    url = new URL(candidate)
  } catch {
    return { _tag: "invalid", reason: "This address is not valid." }
  }

  if (url.protocol === "https:") return { _tag: "navigate", url: url.href }
  if (url.protocol === "http:") {
    return isLoopbackHostname(url.hostname)
      ? { _tag: "navigate", url: url.href }
      : { _tag: "insecure", url: url.href }
  }
  return {
    _tag: "invalid",
    reason: `Magnitude cannot open ${url.protocol.replace(":", "")} addresses.`,
  }
}

export const resolveBrowserNavigation = (input: string): BrowserNavigationTarget => {
  const value = input.trim()
  if (value.length === 0) {
    return { _tag: "invalid", reason: "Enter an address or search." }
  }

  if (value.startsWith("//")) return classifyUrl(`https:${value}`)
  if (HOST_LIKE.test(value)) {
    const hostname = value.startsWith("[")
      ? value.slice(1, value.indexOf("]"))
      : value.split(/[/:?#]/, 1)[0] ?? ""
    return classifyUrl(`${isLoopbackHostname(hostname) ? "http" : "https"}://${value}`)
  }
  if (SCHEME.test(value)) return classifyUrl(value)
  return { _tag: "navigate", url: `${SEARCH_ENDPOINT}${encodeURIComponent(value)}` }
}

export const isAllowedBrowserNavigation = (value: string): boolean => {
  const target = classifyUrl(value)
  return target._tag === "navigate" || target._tag === "insecure"
}

export const browserNavigationFailureMessage = (errorCode: number): string => {
  switch (errorCode) {
    case -7:
    case -118:
      return "The connection timed out."
    case -102:
      return "The server refused the connection."
    case -105:
      return "The server could not be found."
    case -106:
      return "This computer appears to be offline."
    case -200:
    case -202:
      return "This site’s identity could not be verified."
    default:
      return "This page could not be loaded."
  }
}
