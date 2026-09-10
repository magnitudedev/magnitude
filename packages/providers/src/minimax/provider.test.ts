import { describe, expect, it } from "vitest"
import { createMiniMaxProvider, MINIMAX_ENDPOINTS } from "./provider"

describe("MiniMax provider configuration", () => {
  it("uses the global endpoints by default", () => {
    const instance = createMiniMaxProvider({ apiKey: "test-key" })

    expect(instance.provider.id).toBe("minimax")
    expect(instance.authentication._tag).toBe("Configured")
    expect(instance.endpoints).toEqual({
      openaiBaseUrl: "https://api.minimax.io/v1",
      anthropicBaseUrl: "https://api.minimax.io/anthropic",
    })
  })

  it("exposes the regional endpoint pair", () => {
    expect(MINIMAX_ENDPOINTS.cn_zh).toEqual({
      openaiBaseUrl: "https://api.minimaxi.com/v1",
      anthropicBaseUrl: "https://api.minimaxi.com/anthropic",
    })
    expect(createMiniMaxProvider({ apiKey: " ", region: "cn_zh" })).toMatchObject({
      authentication: { _tag: "NotConfigured" },
      endpoints: MINIMAX_ENDPOINTS.cn_zh,
    })
  })
})
