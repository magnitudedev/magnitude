import type { FromServerEncoded, ResponseChunkEncoded } from "@effect/rpc/RpcMessage"

const serverMessageTags: ReadonlySet<string> = new Set([
  "Chunk",
  "Exit",
  "Defect",
  "Pong",
  "ClientProtocolError",
])

export const isChunkMessage = (message: FromServerEncoded): message is ResponseChunkEncoded =>
  message._tag === "Chunk"

export const isFromServerEncoded = (value: unknown): value is FromServerEncoded =>
  typeof value === "object" && value !== null && "_tag" in value
  && typeof value._tag === "string" && serverMessageTags.has(value._tag)

export const isTerminalMessage = (message: FromServerEncoded): boolean =>
  message._tag === "Exit" || message._tag === "Defect" || message._tag === "ClientProtocolError"
