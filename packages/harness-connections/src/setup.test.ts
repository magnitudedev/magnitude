import { Option, Schema } from "effect"
import { parse as parseJsonc } from "jsonc-parser"
import { parse as parseToml } from "smol-toml"
import { parse as parseYaml } from "yaml"
import { describe, expect, it } from "vitest"
import { HARNESS_PRIORITY, ProviderModelIdSchema, ReasoningEffortSchema, harnessCommand } from "@magnitudedev/sdk"
import { HarnessModelSchema } from "./contract"
import { openCodeProviderConfig } from "./connectors/opencode"
import { piProviderConfig } from "./connectors/pi"
import { LOCAL_TOKEN, harnessEndpoints } from "./shared"
import { describeHarnessSetup, type SetupPlatform } from "./setup"

const high = ReasoningEffortSchema.make("high")
const model = Schema.decodeUnknownSync(HarnessModelSchema)({
  id: ProviderModelIdSchema.make("qwen3.5-4b:gguf:q4"),
  name: "Qwen3.5 4B (Q4)",
  description: "A local test model.",
  contextWindow: 65_536,
  maxOutputTokens: 16_384,
  capabilities: { vision: true, tools: true, structuredOutput: true, reasoning: { supported: true, efforts: [high], defaultEffort: high } },
})
const key = "mag-test-network-key"
const origin = "http://192.168.1.20:10100"

/** Every fenced block in a prompt, with the language the prompt labels it with. */
const blocks = (prompt: string) => [...prompt.matchAll(/```(\w+)\n([\s\S]*?)\n```/g)].map(([, format, body]) => ({ format: format!, body: body! }))
const parsed = ({ format, body }: { format: string; body: string }): unknown =>
  format === "json" || format === "json5" ? parseJsonc(body) : format === "yaml" ? parseYaml(body) : parseToml(body)

describe("browser harness setup", () => {
  for (const harness of HARNESS_PRIORITY) for (const platform of ["darwin", "linux", "win32"] as const satisfies readonly SetupPlatform[]) for (const remote of [true, false]) {
    it(`${harness} on ${platform}${remote ? " from another device" : " on the server"}`, () => {
      const setup = describeHarnessSetup({ harness, models: [model], model, platform, origin, key: remote ? Option.some(key) : Option.none() })
      expect(setup.runCommand).toBe(harnessCommand(harness, model.id, platform))
      expect(setup.prompt).toContain(setup.runCommand)
      expect(setup.prompt).toContain(model.id)
      expect(setup.docsUrl).toBe(`https://docs.magnitude.dev/integrations/${harness}`)
      expect(setup.prompt.includes(key)).toBe(remote)
      expect(setup.prompt).toContain(platform === "win32" ? "%USERPROFILE%\\" : "~/")
      expect(setup.prompt).not.toContain(platform === "win32" ? "~/." : "%USERPROFILE%")
      expect(setup.prompt).toContain(platform === "win32" ? "curl.exe" : "curl -s")
      const configs = blocks(setup.prompt)
      expect(configs.length).toBeGreaterThan(0)
      // Every block is valid in its own format and points at this Magnitude, never at loopback.
      for (const block of configs) expect(parsed(block)).toBeTypeOf("object")
      const all = configs.map(block => block.body).join("\n")
      expect(all).toContain(origin)
      expect(all).not.toContain("127.0.0.1")
      if (harness !== "claude-code") expect(setup.prompt.includes(remote ? key : LOCAL_TOKEN)).toBe(true)
    })
  }

  it("carries the model's own limits rather than placeholder values", () => {
    const setup = describeHarnessSetup({ harness: "opencode" as never, models: [model], model, platform: "darwin", origin, key: Option.none() })
    expect(setup.prompt).toContain("65536")
    expect(setup.prompt).toContain("16384")
    expect(setup.prompt).not.toContain("32768")
  })

  it("renders the same provider the one-click connectors write", () => {
    const endpoints = harnessEndpoints(origin)
    const pi = parsed(blocks(describeHarnessSetup({ harness: "pi" as never, models: [model], model, platform: "darwin", origin, key: Option.none() }).prompt)[0]!)
    expect(pi).toEqual({ providers: { magnitude: piProviderConfig([model], endpoints.openai) } })
    const opencode = parsed(blocks(describeHarnessSetup({ harness: "opencode" as never, models: [model], model, platform: "darwin", origin, key: Option.none() }).prompt)[0]!)
    expect(opencode).toEqual({ provider: { magnitude: openCodeProviderConfig([model], endpoints.openai) }, model: `magnitude/${model.id}` })
  })

  it("asks Codex for the key through its environment and Claude Code through its token only from another device", () => {
    const codex = describeHarnessSetup({ harness: "codex" as never, models: [model], model, platform: "linux", origin, key: Option.some(key) })
    expect(parsed(blocks(codex.prompt)[0]!)).toMatchObject({ model_provider: "magnitude", model_providers: { magnitude: { env_key: "MAGNITUDE_API_KEY", base_url: `${origin}/inference/v1/proxies/codex` } } })
    expect(codex.prompt).toContain(`MAGNITUDE_API_KEY=${key}`)
    const remoteClaude = parsed(blocks(describeHarnessSetup({ harness: "claude-code" as never, models: [model], model, platform: "darwin", origin, key: Option.some(key) }).prompt)[0]!)
    expect(remoteClaude).toMatchObject({ env: { ANTHROPIC_AUTH_TOKEN: key, ANTHROPIC_BASE_URL: `${origin}/inference/anthropic/proxies/claude-code` } })
    const localClaude = parsed(blocks(describeHarnessSetup({ harness: "claude-code" as never, models: [model], model, platform: "darwin", origin, key: Option.none() }).prompt)[0]!)
    expect(localClaude).not.toHaveProperty("env.ANTHROPIC_AUTH_TOKEN")
  })
})
