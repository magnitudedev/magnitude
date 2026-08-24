import { describe, expect, it } from "vitest"
import { parse } from "jsonc-parser"
import {
  updateCodexConfig,
  updateOpenCodeConfig,
  updatePiModelsConfig,
  updatePiSettingsConfig,
} from "./connect"

describe("harness connection configuration", () => {
  it("upserts OpenCode and Pi configuration without removing unrelated JSONC settings", () => {
    const source = `{
  // user setting
  "theme": "dark",
  "provider": { "existing": { "name": "Existing" } }
}`
    const openCode = parse(updateOpenCodeConfig(source, "model:gguf:q4"))
    expect(openCode.theme).toBe("dark")
    expect(openCode.provider.existing.name).toBe("Existing")
    expect(openCode.provider.magnitude.options.baseURL).toBe(
      "http://127.0.0.1:10100/inference/v1",
    )
    expect(openCode.model).toBe("magnitude/model:gguf:q4")

    const piModels = parse(updatePiModelsConfig(source, "model:gguf:q4"))
    expect(piModels.theme).toBe("dark")
    expect(piModels.providers.magnitude.models).toEqual([{ id: "model:gguf:q4" }])
    const piSettings = parse(updatePiSettingsConfig(source, "model:gguf:q4"))
    expect(piSettings).toMatchObject({
      theme: "dark",
      defaultProvider: "magnitude",
      defaultModel: "model:gguf:q4",
    })
  })

  it("owns one replaceable Codex provider/profile block and preserves the rest", () => {
    const initial = `approval_policy = "never"

[projects."/workspace"]
trust_level = "trusted"
`
    const first = updateCodexConfig(initial, "first:gguf:q4")
    const second = updateCodexConfig(first, "second:gguf:q8")
    expect(second).toContain('approval_policy = "never"')
    expect(second).toContain('[projects."/workspace"]')
    expect(second).toContain('base_url = "http://127.0.0.1:10100/inference/v1"')
    expect(second).toContain('wire_api = "chat"')
    expect(second).toContain('model = "second:gguf:q8"')
    expect(second).not.toContain("first:gguf:q4")
    expect(second.match(/\[model_providers\.magnitude\]/g)).toHaveLength(1)
  })
})
