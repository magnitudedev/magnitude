import { describe, expect, it } from "vitest"
import type { ModelLocation } from "@magnitudedev/sdk"
import { shownModelLocationPaths } from "./model-locations"

const ollama = (paths: ModelLocation["paths"], state: ModelLocation["state"]): ModelLocation =>
  ({ kind: "Ollama", paths, enabled: true, automatic: true, state })

describe("shownModelLocationPaths", () => {
  it("shows the directories found or unreadable and leaves out the missing ones", () => {
    expect(shownModelLocationPaths(ollama([
      { path: "/home/ada/.ollama/models", state: "Found" },
      { path: "/usr/share/ollama/.ollama/models", state: "NotFound" },
      { path: "/var/lib/ollama", state: "Unreadable" },
    ], "Found"))).toEqual([
      { path: "/home/ada/.ollama/models", state: "Found" },
      { path: "/var/lib/ollama", state: "Unreadable" },
    ])
  })

  it("shows every directory checked when none exists", () => {
    const paths: ModelLocation["paths"] = [
      { path: "/home/ada/.ollama/models", state: "NotFound" },
      { path: "/var/lib/ollama", state: "NotFound" },
    ]
    expect(shownModelLocationPaths(ollama(paths, "NotFound"))).toEqual(paths)
  })
})
