import * as Diff from "diff"

const utf8Decoder = new TextDecoder("utf-8", { fatal: true })

function isBinary(bytes: Uint8Array): boolean {
  if (bytes.includes(0)) return true
  try {
    utf8Decoder.decode(bytes)
    return false
  } catch {
    return true
  }
}

export function createContentPatch(
  filePath: string,
  oldBytes: Uint8Array | null,
  newBytes: Uint8Array | null,
): string {
  const oldContent = oldBytes ?? new Uint8Array()
  const newContent = newBytes ?? new Uint8Array()

  if (isBinary(oldContent) || isBinary(newContent)) {
    return [
      `diff --git a/${filePath} b/${filePath}`,
      `Binary files a/${filePath} and b/${filePath} differ`,
      "",
    ].join("\n")
  }

  const decoder = new TextDecoder()
  return Diff.createPatch(
    filePath,
    decoder.decode(oldContent),
    decoder.decode(newContent),
    "a/" + filePath,
    "b/" + filePath,
  )
}
