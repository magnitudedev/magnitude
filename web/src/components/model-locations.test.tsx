import { renderToStaticMarkup } from "react-dom/server"
import { describe, expect, it } from "vitest"
import { Option } from "effect"
import type { DirectoryListing, ModelLocation } from "@magnitudedev/sdk"
import { ModelLocationsManager, modelLocationsSummary } from "./model-locations"
import { DirectoryEntries } from "./folder-picker"
import { candidateErrorNotice, modelLocationErrorNotice } from "./candidate-errors"
import { DirectoryNotFound, InvalidModelReference, ModelSourceUnavailable } from "@magnitudedev/sdk"

const locations: ModelLocation[] = [
  { kind: "HuggingFaceCache", path: "/Users/me/.cache/huggingface/hub", enabled: true, automatic: true, state: "Found" },
  { kind: "LmStudio", path: "/Users/me/.lmstudio/models", enabled: false, automatic: true, state: "Found" },
  { kind: "Ollama", path: "/Users/me/.ollama/models", enabled: true, automatic: true, state: "NotFound" },
  { kind: "Folder", path: "/Volumes/External/models", enabled: true, automatic: false, state: "Found" },
]

describe("model locations", () => {
  const manager = (entries: readonly ModelLocation[]) => renderToStaticMarkup(<ModelLocationsManager locations={entries} busy={false} notice={undefined} onToggle={() => {}} onRemove={() => {}} onAdd={() => {}} />)
  const html = manager(locations)
  it("summarizes the locations found and on, then the folder count, in one row", () => {
    expect(modelLocationsSummary(locations)).toBe("Hugging Face cache · 1 folder")
    expect(modelLocationsSummary([...locations, { ...locations[3]!, path: "/models" }])).toBe("Hugging Face cache · 2 folders")
    expect(modelLocationsSummary([{ ...locations[1]!, enabled: true }])).toBe("LM Studio")
    expect(modelLocationsSummary([locations[2]!])).toBe("Magnitude isn’t looking anywhere else for models")
  })
  it("lists found automatic locations with their path and an on/off switch, and tucks away the ones not found", () => {
    expect(html).toContain("Hugging Face cache")
    expect(html).toContain("/Users/me/.cache/huggingface/hub")
    expect(html).toContain('aria-label="Look for models in LM Studio"')
    expect(html).not.toContain('aria-label="Look for models in Ollama"')
    expect(html.match(/role="switch"/g)).toHaveLength(2)
    expect(html).toMatch(/<details[^>]*><summary[^>]*>Not found on this computer \(1\)<\/summary>.*Ollama.*\/Users\/me\/.ollama\/models/)
  })
  it("lists your folders with Remove under the Your folders heading, beside Add folder…", () => {
    expect(html).toContain(">models<")
    expect(html).toContain('aria-label="Remove /Volumes/External/models"')
    expect(html.indexOf("Your folders")).toBeLessThan(html.indexOf("Add folder…"))
    expect(html.indexOf("Add folder…")).toBeLessThan(html.indexOf("Remove /Volumes"))
    expect(html.indexOf("Not found on this computer")).toBeLessThan(html.indexOf("Your folders"))
  })
  it("keeps locations it can't read in view with what to do, and leaves them out of the summary", () => {
    const unreadable: ModelLocation[] = [
      locations[0]!,
      { kind: "Ollama", path: "/usr/share/ollama/.ollama/models", enabled: true, automatic: true, state: "Unreadable" },
      { kind: "Folder", path: "/srv/models", enabled: true, automatic: false, state: "Unreadable" },
    ]
    const html = manager(unreadable)
    expect(modelLocationsSummary(unreadable)).toBe("Hugging Face cache · 1 folder")
    expect(html).toContain('aria-label="Look for models in Ollama"')
    expect(html).toContain("Can’t read · /usr/share/ollama/.ollama/models")
    expect(html).toContain("A system Ollama service keeps its models in a folder that belongs to the ollama user: give your account read access to it, or set OLLAMA_MODELS to a folder you can read.")
    expect(html).toContain("Can’t read · /srv/models")
    expect(html).toContain("Give your account read access to it.")
    expect(html).not.toContain("Not found on this computer")
  })
  it("explains a first folder before any exist", () => {
    expect(manager(locations.slice(0, 1))).toContain("Add a folder of GGUF files no other app manages")
    expect(manager(locations.slice(0, 1))).not.toContain("<details")
  })
  it("words folder and reference failures without service text", () => {
    expect(modelLocationErrorNotice(new DirectoryNotFound({ path: "/x" })).title).toBe("This folder doesn’t exist")
    expect(candidateErrorNotice(new InvalidModelReference({ reference: "x", message: "PRIVATE" }))).toMatchObject({ title: "Magnitude can’t open this model", retryable: false })
    expect(candidateErrorNotice(new ModelSourceUnavailable({ message: "PRIVATE", retryable: true }))).toMatchObject({ title: "Couldn’t reach Hugging Face", retryable: true })
    expect(JSON.stringify(candidateErrorNotice(new ModelSourceUnavailable({ message: "PRIVATE", retryable: false })))).not.toContain("PRIVATE")
    expect(candidateErrorNotice(new Error("transport")).title).toBe("Couldn’t reach the Magnitude service")
  })
})

describe("GGUF file picker", () => {
  const listing: DirectoryListing = {
    path: "/Users/me/models", parent: Option.some("/Users/me"),
    directories: [{ name: "drafts", path: "/Users/me/models/drafts" }],
    files: [{ name: "qwen.gguf", path: "/Users/me/models/qwen.gguf", sizeBytes: 23_400_000_000 }],
  }
  it("lists GGUF files with their sizes and marks the chosen one", () => {
    const html = renderToStaticMarkup(<DirectoryEntries kind="gguf" listing={listing} selectedFile={Option.some("/Users/me/models/qwen.gguf")} onOpen={() => {}} onSelectFile={() => {}} />)
    expect(html).toContain("Parent folder")
    expect(html).toContain("drafts")
    expect(html).toContain("qwen.gguf")
    expect(html).toContain("23.4 GB")
    expect(html).toContain('aria-pressed="true"')
  })
  it("shows only folders when choosing a folder", () => {
    const html = renderToStaticMarkup(<DirectoryEntries kind="folder" listing={listing} selectedFile={Option.none()} onOpen={() => {}} onSelectFile={() => {}} />)
    expect(html).not.toContain("qwen.gguf")
    expect(renderToStaticMarkup(<DirectoryEntries kind="gguf" listing={{ ...listing, directories: [], files: [] }} selectedFile={Option.none()} onOpen={() => {}} onSelectFile={() => {}} />)).toContain("No folders or GGUF files inside")
  })
})
