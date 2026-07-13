import { describe, expect, it } from "vitest"
import {
  MagnitudeRpcs,
  STREAM_LIVENESS_TIMEOUT_MS,
} from "@magnitudedev/protocol"
import {
  acnResidentStreamTags,
  acnResidentStreamPolicy,
  isEncodedHeartbeat,
} from "./acn-stream-policy"

describe("acnResidentStreamTags", () => {
  it("contains stream RPCs (WatchFile, StreamDisplayView)", () => {
    expect(acnResidentStreamTags.has("WatchFile")).toBe(true)
    expect(acnResidentStreamTags.has("StreamDisplayView")).toBe(true)
  })

  it("does not contain unary RPCs (CheckFileExists)", () => {
    expect(acnResidentStreamTags.has("CheckFileExists")).toBe(false)
    expect(acnResidentStreamTags.has("ListFiles")).toBe(false)
  })

  it("every tag corresponds to an Rpc with stream: true in MagnitudeRpcs", () => {
    for (const tag of acnResidentStreamTags) {
      const rpc = MagnitudeRpcs.requests.get(tag)
      expect(rpc).toBeDefined()
    }
  })
})

describe("isEncodedHeartbeat", () => {
  it("identifies { _tag: 'heartbeat' } as a heartbeat", () => {
    expect(isEncodedHeartbeat({ _tag: "heartbeat" })).toBe(true)
  })

  it("rejects non-heartbeat values", () => {
    expect(isEncodedHeartbeat({ _tag: "changed", path: "/x" })).toBe(false)
    expect(isEncodedHeartbeat({ event: "changed", path: "/x" })).toBe(false)
    expect(isEncodedHeartbeat(null)).toBe(false)
    expect(isEncodedHeartbeat(undefined)).toBe(false)
    expect(isEncodedHeartbeat("heartbeat")).toBe(false)
    expect(isEncodedHeartbeat(42)).toBe(false)
    expect(isEncodedHeartbeat({})).toBe(false)
  })
})

describe("acnResidentStreamPolicy", () => {
  it("isResident returns true for a known stream tag and false for a unary tag", () => {
    expect(acnResidentStreamPolicy.isResident("WatchFile")).toBe(true)
    expect(acnResidentStreamPolicy.isResident("StreamDisplayView")).toBe(true)
    expect(acnResidentStreamPolicy.isResident("CheckFileExists")).toBe(false)
  })

  it("isHeartbeatChunk delegates to isEncodedHeartbeat", () => {
    expect(acnResidentStreamPolicy.isHeartbeatChunk({ _tag: "heartbeat" })).toBe(true)
    expect(acnResidentStreamPolicy.isHeartbeatChunk({ _tag: "changed" })).toBe(false)
    expect(acnResidentStreamPolicy.isHeartbeatChunk(null)).toBe(false)
  })

  it("livenessTimeoutMs equals STREAM_LIVENESS_TIMEOUT_MS from protocol", () => {
    expect(acnResidentStreamPolicy.livenessTimeoutMs).toBe(STREAM_LIVENESS_TIMEOUT_MS)
  })

  it("isRelinquishExit returns true for a clean success exit", () => {
    const cleanExit = {
      _tag: "Success" as const,
      value: undefined,
      previousExit: undefined,
      meta: undefined,
    }
    expect(acnResidentStreamPolicy.isRelinquishExit(cleanExit)).toBe(true)
  })

  it("isRelinquishExit returns false for a domain failure exit", () => {
    const failExit = {
      _tag: "Failure" as const,
      cause: { _tag: "Fail" as const, error: { _tag: "SessionNotFound", sessionId: "s" } },
      previousExit: undefined,
      meta: undefined,
    }
    expect(acnResidentStreamPolicy.isRelinquishExit(failExit)).toBe(false)
  })
})
