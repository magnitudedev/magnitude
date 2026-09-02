import { describe, expect, it } from "vitest"
import { documentationTopics } from "../agent-docs/topics"
import {
  renderDocumentationDirectory,
  resolveDocumentationCommand,
} from "./docs"

describe("Magnitude documentation", () => {
  it("renders the complete topic directory in registry order", () => {
    const directory = renderDocumentationDirectory()

    expect(directory).toContain("Usage: magnitude docs <topic-id>")
    for (const topic of documentationTopics) {
      expect(directory).toContain(topic.id)
      expect(directory).toContain(topic.description)
    }
  })

  it("returns bundled Markdown with exactly one trailing newline", () => {
    const result = resolveDocumentationCommand("custom-endpoints")

    expect(result._tag).toBe("Success")
    if (result._tag === "Failure") return
    expect(result.output).toContain("# Custom endpoints")
    expect(result.output).toContain("OPENROUTER_API_KEY")
    expect(result.output).toMatch(/[^\n]\n$/)
  })

  it("publishes the complete agent-guided onboarding workflow", () => {
    const result = resolveDocumentationCommand("onboarding")

    expect(result._tag).toBe("Success")
    if (result._tag === "Failure") return
    expect(result.output).toContain("# Agent-guided Magnitude onboarding")
    expect(result.output).toContain("interactive conversation")
    expect(result.output).toContain("Welcome questions")
    expect(result.output).toContain("Use `magnitude docs` to answer")
    expect(result.output).toContain("magnitude service install")
    expect(result.output).not.toContain("magnitude --version")
    expect(result.output).toContain("magnitude catalog status")
    expect(result.output).toContain("magnitude catalog recommendations --preference balanced")
    expect(result.output).not.toContain("output is stable")
    expect(result.output).toContain("do not interpret that as a request for the extremes")
    expect(result.output).toContain("Treat `fastest` and `smartest` as explicit extremes")
    expect(result.output).toContain("clear, use `faster` or `smarter` instead")
    expect(result.output).toContain("## 3. Install the chosen model")
    expect(result.output).toContain("## 4. Load the installed model")
    expect(result.output).not.toContain("residency slot")
    expect(result.output).not.toContain("identify the harness")
    expect(result.output).not.toContain("rediscover your own harness")
    expect(result.output).not.toContain("genuinely ambiguous")
    expect(result.output).toContain("connected instead or in addition")
    expect(result.output).toContain("Adjust the polling interval")
    expect(result.output).not.toContain("Poll `models status` about every")
    expect(result.output).not.toContain("next launch")
    expect(result.output).toContain("selects the chosen model in its")
    expect(result.output).not.toContain("default for ordinary new sessions")
    expect(result.output).not.toContain("by default")
    expect(result.output).not.toContain("Describe `--set-model`")
    expect(result.output).toContain("instructions for using the Magnitude CLI to manage local models")
    expect(result.output).toContain("The loaded model is selected automatically")
    expect(result.output).toContain("Pi:** switch in place")
    expect(result.output).toContain("| OpenCode | `opencode` | Exit the running process")
    expect(result.output).toContain("restart the harness process")
    expect(result.output).not.toContain("/models")
    expect(result.output).not.toContain("/new")
    expect(result.output).not.toContain("For example:")
    expect(result.output).not.toContain("Also tell them")
    expect(result.output).not.toContain("tell the user")
    expect(result.output).not.toContain("tell them")
    expect(result.output).toContain("ordinary command name, not an absolute executable")
    expect(result.output).not.toContain("--json")
    expect(result.output).toContain("--install-skill")
  })

  it("publishes self-contained speculative method guidance", () => {
    const result = resolveDocumentationCommand("speculative-methods")

    expect(result._tag).toBe("Success")
    if (result._tag === "Failure") return
    expect(result.output).toContain("# Speculative decoding methods")
    expect(result.output).toContain(
      "None (usually slowest) -> MTP -> DFlash -> DSpark (usually fastest)",
    )
    expect(result.output).toContain("Magnitude then activates the method automatically")
    expect(result.output).toContain("typical ordering, not a guarantee")
  })

  it("publishes recommendation methodology and reference points", () => {
    const result = resolveDocumentationCommand("recommendations")

    expect(result._tag).toBe("Success")
    if (result._tag === "Failure") return
    expect(result.output).toContain("# Model recommendations")
    expect(result.output).toContain("Artificial Analysis Intelligence Index")
    expect(result.output).toContain("normal requests to lean toward speed or intelligence")
    expect(result.output).toContain("`fastest` and `smartest` are extremes")
    expect(result.output).toContain("`smartest` prioritizes intelligence and gives speed only")
    expect(result.output).toContain("shorter and longer contexts")
    expect(result.output).toContain("useful reference points")
    expect(result.output).toContain("magnitude docs speculative-methods")
    expect(result.output).toContain("GPT-5.6 Sol")
    expect(result.output).toContain("Claude Opus 5")
    expect(result.output).not.toContain("utility =")
    expect(result.output).not.toContain("25K")
    expect(result.output).not.toContain("50K")
    expect(result.output).not.toContain("75K")
  })

  it("rejects unknown topic IDs and lists the available IDs", () => {
    const result = resolveDocumentationCommand("CUSTOM-ENDPOINTS")

    expect(result._tag).toBe("Failure")
    if (result._tag === "Success") return
    expect(result.error).toContain(
      "Unknown Magnitude documentation topic: CUSTOM-ENDPOINTS",
    )
    expect(result.error).toContain("custom-endpoints")
  })
})
