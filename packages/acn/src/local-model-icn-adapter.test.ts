import { describe, expect, it } from "vitest"
import { Effect, Option, Schema } from "effect"
import { DownloadAttempt as NativeDownloadAttemptSchema } from "@magnitudedev/icn-protocol/schemas"
import { downloadAttemptFromIcn } from "./local-model-icn-adapter"

describe("local model ICN adapter", () => {
  it("projects a decoded native download rate into the protocol representation", async () => {
    const attempt = Schema.decodeUnknownSync(NativeDownloadAttemptSchema)({
      _tag: "Downloading",
      id: "download_test",
      packageId: "package_test",
      stage: "downloading",
      completedBytes: 4_000,
      totalBytes: 10_000,
      bytesPerSecond: 2_000,
    })

    const projected = await Effect.runPromise(downloadAttemptFromIcn(attempt))

    expect(projected).toEqual({
      _tag: "Downloading",
      id: "download_test",
      packageId: "package_test",
      stage: "downloading",
      completedBytes: 4_000,
      totalBytes: 10_000,
      bytesPerSecond: Option.some(2_000),
    })
  })

  it("projects a missing native download rate as None", async () => {
    const attempt = Schema.decodeUnknownSync(NativeDownloadAttemptSchema)({
      _tag: "Downloading",
      id: "download_test",
      packageId: "package_test",
      stage: "downloading",
      completedBytes: 4_000,
      totalBytes: 10_000,
    })

    const projected = await Effect.runPromise(downloadAttemptFromIcn(attempt))

    expect(projected._tag).toBe("Downloading")
    if (projected._tag === "Downloading") {
      expect(projected.bytesPerSecond).toEqual(Option.none())
    }
  })
})
