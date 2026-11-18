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

  it("lists the headless topic first", () => {
    expect(documentationTopics[0]?.id).toBe("headless")
    const rows = renderDocumentationDirectory().split("\n").filter((line) => /^\S+\s{2,}\S/.test(line))
    expect(rows[1]?.startsWith("headless ")).toBe(true)
  })

  it("documents background serving through the magnitude command on every OS", () => {
    const result = resolveDocumentationCommand("headless")

    expect(result._tag).toBe("Success")
    if (result._tag === "Failure") return
    expect(result.output).toContain("Ask your user before setting this up.")
    expect(result.output).toContain("<string>$HOME/.magnitude/bin/magnitude</string><string>serve</string>")
    expect(result.output).toContain("launchctl bootstrap gui/$(id -u)")
    expect(result.output).toContain("<key>UserName</key>")
    expect(result.output).toContain("ExecStart=%h/.magnitude/bin/magnitude serve")
    expect(result.output).toContain("sudo loginctl enable-linger $USER")
    expect(result.output).toContain("New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME")
    expect(result.output).toContain("-ExecutionTimeLimit ([TimeSpan]::Zero)")
    expect(result.output).toContain("Unregister-ScheduledTask")
    expect(result.output).toContain("autoDownloadUpdates")
    expect(result.output).not.toMatch(/cli\/(current|<version>|\d)[^`\n]*magnitude serve/)
  })

  it("no longer documents server mode", () => {
    for (const topic of documentationTopics) {
      expect(topic.markdown).not.toContain("server setup")
      expect(topic.markdown).not.toContain("server remove")
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
    if (result._tag === "Success") expect(result.output).toContain("Use Desktop or `magnitude serve` to run inference.")
  })

  it("publishes self-contained speculative method guidance", () => {
    const result = resolveDocumentationCommand("speculative-methods")

    expect(result._tag).toBe("Success")
    if (result._tag === "Failure") return
    expect(result.output).toContain("# Speculative decoding methods")
    expect(result.output).toContain(
      "None (usually slowest) -> MTP -> DFlash -> DSpark -> LiLiCorr (usually fastest)",
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
    expect(result.output).toContain("estimated to decode at 20 tokens per second or more")
    expect(result.output).toContain("`--country US` recommends only")
    expect(result.output).toContain("each assessed context")
    expect(result.output).toContain("useful reference points")
    expect(result.output).toContain("magnitude docs speculative-methods")
    expect(result.output).toContain("GPT-5.6 Sol")
    expect(result.output).toContain("Claude Opus 5")
    expect(result.output).not.toContain("utility =")
    expect(result.output).not.toContain("25K")
    expect(result.output).not.toContain("50K")
    expect(result.output).not.toContain("75K")
  })

  it("documents custom models, drafters, and model locations", () => {
    const result = resolveDocumentationCommand("custom-models")

    expect(result._tag).toBe("Success")
    if (result._tag === "Failure") return
    for (const command of ["models search", "models show", "models pull", "models rm", "models drafter", "models hide", "models folders"]) {
      expect(result.output).toContain(command)
    }
    const cli = resolveDocumentationCommand("cli")
    expect(cli._tag === "Success" && cli.output).toContain("accepts `--json`")
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
