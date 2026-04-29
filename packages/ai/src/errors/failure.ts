export interface HttpConnectionFailure {
  readonly status: number
  readonly headers: Headers
  readonly body: string
}

export type StreamFailure =
  | { readonly _tag: "ReadFailure"; readonly cause: Error }
  | { readonly _tag: "SseParseFailure"; readonly payload: string }
  | { readonly _tag: "ChunkDecodeFailure"; readonly payload: string; readonly cause: Error }
