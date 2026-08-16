import { describe, expect, test } from "vitest"
import { Option } from "effect"
import {
  decodeDesktopAcnEnsureEvent,
  encodeDesktopAcnEnsureEvent,
} from "./desktop-rpc"

describe("desktop ACN ensure bridge", () => {
  test("restores Effect options after Electron structured cloning", () => {
    const encoded = encodeDesktopAcnEnsureEvent({
      _tag: "Observation",
      observation: {
        _tag: "Installing",
        phase: "DownloadingInferenceEngine",
        plan: {
          daemonBytes: 1,
          inferenceEngineBytes: 100,
          inferenceEngineBytesExact: true,
        },
        progress: Option.some({
          completed: 50,
          totalBytes: 100,
          unit: "Bytes",
          attempt: Option.some(1),
        }),
      },
    })

    const decoded = decodeDesktopAcnEnsureEvent(structuredClone(encoded))
    expect(decoded._tag).toBe("Observation")
    if (
      decoded._tag !== "Observation" ||
      decoded.observation._tag !== "Installing"
    ) {
      throw new Error("expected an installing observation")
    }
    expect(Option.getOrThrow(decoded.observation.progress)).toEqual({
      completed: 50,
      totalBytes: 100,
      unit: "Bytes",
      attempt: Option.some(1),
    })
  })
})
