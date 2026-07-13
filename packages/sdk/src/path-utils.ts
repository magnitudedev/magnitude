export function normalizeReferencedPath(refPath: string): string | null {
  let value = refPath.trim()
  if (!value) return null

  value = value.replace(/\\/g, "/")
  value = value.replace(/^\.\/+/, "")

  const parts: string[] = []
  for (const part of value.split("/")) {
    if (!part || part === ".") continue
    if (part === "..") {
      if (parts.length === 0) return null
      parts.pop()
      continue
    }
    parts.push(part)
  }

  return parts.length > 0 ? parts.join("/") : null
}
