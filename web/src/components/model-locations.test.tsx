import { renderToStaticMarkup } from "react-dom/server"
import { describe, expect, it } from "vitest"
import { Option } from "effect"
import type { DirectoryListing, ModelLocation } from "@magnitudedev/sdk"
import { ModelLocationRows } from "./model-locations"
import { DirectoryEntries } from "./folder-picker"
import { candidateErrorNotice, modelLocationErrorNotice } from "./candidate-errors"
import { DirectoryNotFound, InvalidModelReference, ModelSourceUnavailable } from "@magnitudedev/sdk"

const locations: ModelLocation[] = [
  { kind: "HuggingFaceCache", path: "/Users/me/.cache/huggingface/hub", enabled: true, automatic: true, found: true },
  { kind: "LmStudio", path: "/Users/me/.lmstudio/models", enabled: false, automatic: true, found: true },
  { kind: "Ollama", path: "/Users/me/.ollama/models", enabled: true, automatic: true, found: false },
  { kind: "Folder", path: "/Volumes/External/models", enabled: true, automatic: false, found: true },
]

describe("model locations", () => {
  const html = renderToStaticMarkup(<ModelLocationRows locations={locations} busy={false} onToggle={() => {}} onRemove={() => {}} onAdd={() => {}} />)
  it("lists automatic locations with their path and an on/off switch when found", () => {
    expect(html).toContain("Hugging Face cache")
    expect(html).toContain("/Users/me/.cache/huggingface/hub")
    expect(html).toContain('aria-label="Look for models in LM Studio"')
    expect(html).toContain("Not found · /Users/me/.ollama/models")
    expect(html).not.toContain('aria-label="Look for models in Ollama"')
    expect(html.match(/role="switch"/g)).toHaveLength(2)
  })
  it("lists your folders with Remove, then Add folder…", () => {
    expect(html).toContain(">models<")
    expect(html).toContain('aria-label="Remove /Volumes/External/models"')
    expect(html).toContain("Add folder…")
    expect(html.indexOf("Remove /Volumes")).toBeLessThan(html.indexOf("Add folder…"))
  })
  it("explains a first folder before any exist", () => {
    expect(renderToStaticMarkup(<ModelLocationRows locations={locations.slice(0, 1)} busy onToggle={() => {}} onRemove={() => {}} onAdd={() => {}} />)).toContain("Add a folder of GGUF files no other app manages")
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
