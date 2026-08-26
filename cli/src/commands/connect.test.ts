import { describe, expect, it } from "vitest"
import { parse } from "jsonc-parser"
import {
  updateCodexConfig,
  captureClaudeConnectionState,
  restoreClaudeConfig,
  updateClaudeConfig,
  validateClaudeEnvironment,
  validateClaudeCodeVersion,
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

  it("owns only Magnitude's Claude Code settings", () => {
    const source = `{
  "theme": "dark",
  "env": { "EXISTING": "value" }
}`
    const connected = updateClaudeConfig(source)
    const state = captureClaudeConnectionState(source, "/tmp/settings.json")
    expect(parse(connected)).toMatchObject({
      theme: "dark",
      env: {
        EXISTING: "value",
        ANTHROPIC_BASE_URL:
          "http://127.0.0.1:10100/inference/anthropic",
        CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY: "1",
      },
    })
    expect(parse(restoreClaudeConfig(connected, state))).toEqual({
      theme: "dark",
      env: { EXISTING: "value" },
    })
  })

  it("rejects JSONC in Claude settings because Claude requires strict JSON", () => {
    expect(() => updateClaudeConfig(`{ // comment\n }`)).toThrow(
      "strict JSON object",
    )
  })

  it("restores prior Claude values only while Magnitude still owns them", () => {
    const source = JSON.stringify({
      env: {
        ANTHROPIC_BASE_URL: "https://user-gateway.example",
      },
    })
    const state = captureClaudeConnectionState(source, "/tmp/settings.json")
    const connected = updateClaudeConfig(source)
    expect(JSON.parse(restoreClaudeConfig(connected, state))).toEqual({
      env: { ANTHROPIC_BASE_URL: "https://user-gateway.example" },
    })

    const edited = JSON.stringify({
      env: {
        ANTHROPIC_BASE_URL: "https://later-user-edit.example",
        CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY: "1",
      },
    })
    expect(JSON.parse(restoreClaudeConfig(edited, state))).toEqual({
      env: {
        ANTHROPIC_BASE_URL: "https://later-user-edit.example",
      },
    })
  })

  it("requires a Claude Code release with opt-in gateway discovery", () => {
    expect(validateClaudeCodeVersion("2.1.129 (Claude Code)")).toBe("2.1.129")
    expect(validateClaudeCodeVersion("claude 2.2.0")).toBe("2.2.0")
    expect(() => validateClaudeCodeVersion("2.1.128 (Claude Code)")).toThrow("2.1.129 or later")
    expect(() => validateClaudeCodeVersion("unknown")).toThrow("Unable to parse")
  })

  it("rejects shell overrides that would defeat the persistent Claude connection", () => {
    expect(() => validateClaudeEnvironment({
      ANTHROPIC_BASE_URL: "https://other.example",
    })).toThrow("current environment overrides ANTHROPIC_BASE_URL")
    expect(() => validateClaudeEnvironment({
      CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC: "1",
    })).toThrow("disables gateway discovery")
    expect(() => validateClaudeEnvironment({})).not.toThrow()
  })
})
