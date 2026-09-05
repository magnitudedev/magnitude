import { describe, expect, test } from "vitest"
import { Option } from "effect"
import {
  decodeDesktopServiceStartProgress,
  encodeDesktopServiceStartProgress,
} from "./desktop-rpc"

describe("desktop ACN ensure bridge", () => {
  test("restores Effect options after Electron structured cloning", () => {
    const encoded = encodeDesktopServiceStartProgress({
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
    })

    const decoded = decodeDesktopServiceStartProgress(structuredClone(encoded))
    expect(decoded._tag).toBe("Installing")
    if (decoded._tag !== "Installing") {
      throw new Error("expected an installing observation")
    }
    expect(Option.getOrThrow(decoded.progress)).toEqual({
      completed: 50,
      totalBytes: 100,
      unit: "Bytes",
      attempt: Option.some(1),
    })
  })
})
