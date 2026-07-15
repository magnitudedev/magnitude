import { describe, expect, test } from "vitest"

const read = (path: string) => Bun.file(new URL(path, import.meta.url)).text()

describe("local inference client-state boundary", () => {
  test("keeps the CLI hook as pure domain composition", async () => {
    const source = await read("./use-local-inference-onboarding.ts")

    expect(source).toContain("useLocalInferenceState")
    expect(source).not.toMatch(/\buseState\b/)
    expect(source).not.toContain("RpcClient")
    expect(source).not.toContain("Reactivity.invalidate")
    expect(source).not.toMatch(/set(?:Busy|Error|Progress|OperationId|UsageSnapshot)/)
  })

  test("keeps the raw download stream in client-common as invalidation only", async () => {
    const source = await read("../../../../packages/client-common/src/hooks/use-local-inference-state.ts")

    expect(source).toContain("GetLocalModelDownloadProgress")
    expect(source).toContain("SubscribeLocalModelDownload")
    expect(source).toContain("Reactivity.invalidate")
    expect(source).not.toMatch(/set(?:Progress|OperationId|UsageSnapshot)/)
  })
})
