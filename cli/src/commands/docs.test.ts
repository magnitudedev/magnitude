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
    expect(result.output).toContain("magnitude catalog list --json")
    expect(result.output).toContain("--install-skill")
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
