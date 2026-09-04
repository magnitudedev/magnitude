import { Effect } from "effect"
import { describe, expect, it } from "vitest"
import { decodePiDevelopmentStatus } from "./dev-pi"

describe("Pi development status decoding", () => {
  it("accepts initializing and ready model status envelopes", async () => {
    const initializing = await Effect.runPromise(decodePiDevelopmentStatus(JSON.stringify({
      schemaVersion: 1,
      command: "models.status",
      ok: true,
      data: { state: "initializing", models: [] },
    })))
    expect(initializing.ok).toBe(true)
    if (initializing.ok) expect(initializing.data.state).toBe("initializing")

    const ready = await Effect.runPromise(decodePiDevelopmentStatus(JSON.stringify({
      schemaVersion: 1,
      command: "models.status",
      ok: true,
      data: {
        state: "ready",
        models: [{ modelId: "qwen3.6-35b-a3b:gguf:q6", installation: "installed" }],
      },
    })))
    expect(ready.ok).toBe(true)
    if (ready.ok) expect(ready.data.models).toHaveLength(1)
  })

  it("preserves the CLI failure message", async () => {
    const error = await Effect.runPromiseExit(decodePiDevelopmentStatus(JSON.stringify({
      schemaVersion: 1,
      command: "models.status",
      ok: false,
      error: { message: "Magnitude service is not running" },
    })))
    expect(error._tag).toBe("Failure")
    expect(String(error)).toContain("Magnitude service is not running")
  })

  it("rejects malformed or incompatible responses", async () => {
    const error = await Effect.runPromiseExit(decodePiDevelopmentStatus("not json"))
    expect(error._tag).toBe("Failure")
    expect(String(error)).toContain("incompatible model status response")
  })
})
