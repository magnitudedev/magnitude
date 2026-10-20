/**
 * Accept only local loopback URLs for embedded Open WebUI.
 *
 * This prevents the optional Chat iframe from becoming an arbitrary remote
 * content embed. It does not bypass the embedded server's frame-ancestors or
 * X-Frame-Options policies; those must be configured and tested separately.
 */
export function normalizeLocalEmbedUrl(value: string): string | null {
  try {
    const url = new URL(value.trim())
    const loopbackHosts = new Set(["localhost", "127.0.0.1", "[::1]"])
    if (url.protocol !== "http:" && url.protocol !== "https:") return null
    if (!loopbackHosts.has(url.hostname.toLowerCase())) return null
    if (url.username || url.password) return null
    url.hash = ""
    return url.toString()
  } catch {
    return null
  }
}
