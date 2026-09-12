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

  it("keeps onboarding in the desktop application", () => {
    expect(resolveDocumentationCommand("onboarding")._tag).toBe("Failure")
    const result = resolveDocumentationCommand("cli")
    expect(result._tag).toBe("Success")
    if (result._tag === "Success") expect(result.output).toContain("Onboarding lives in the Magnitude desktop app")
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
