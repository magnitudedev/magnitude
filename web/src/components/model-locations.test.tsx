import { renderToStaticMarkup } from "react-dom/server"
import { describe, expect, it } from "vitest"
import { Option } from "effect"
import type { DirectoryListing, ModelLocation } from "@magnitudedev/sdk"
import { ModelLocationsManager, modelLocationsSummary } from "./model-locations"
import { DirectoryEntries, choosePath, type NativeOpenPath } from "./folder-picker"
import { candidateErrorNotice, modelLocationErrorNotice } from "./candidate-errors"
import { DirectoryNotFound, DrafterNotUsable, InvalidModelReference, ModelRepositoryNotFound, ModelSourceUnavailable } from "@magnitudedev/sdk"

const locations: ModelLocation[] = [
  { kind: "HuggingFaceCache", paths: [
    { path: "/Volumes/hf/hub", state: "Found" },
    { path: "/Users/me/.cache/huggingface/hub", state: "Found" },
  ], enabled: true, automatic: true, state: "Found" },
  { kind: "LmStudio", paths: [
    { path: "/Users/me/.lmstudio/models", state: "Found" },
    { path: "/Users/me/.cache/lm-studio/models", state: "NotFound" },
  ], enabled: false, automatic: true, state: "Found" },
  { kind: "Ollama", paths: [{ path: "/Users/me/.ollama/models", state: "NotFound" }], enabled: true, automatic: true, state: "NotFound" },
  { kind: "Folder", paths: [{ path: "/Volumes/External/models", state: "Found" }], enabled: true, automatic: false, state: "Found" },
]

describe("model locations", () => {
  const manager = (entries: readonly ModelLocation[]) => renderToStaticMarkup(<ModelLocationsManager locations={entries} busy={false} notice={undefined} onToggle={() => {}} onRemove={() => {}} onAdd={() => {}} />)
  const html = manager(locations)
  it("summarizes the locations found and on, then the folder count, in one row", () => {
    expect(modelLocationsSummary(locations)).toBe("Hugging Face cache · 1 folder")
    expect(modelLocationsSummary([...locations, { ...locations[3]!, paths: [{ path: "/models", state: "Found" }] }])).toBe("Hugging Face cache · 2 folders")
    expect(modelLocationsSummary([{ ...locations[1]!, enabled: true }])).toBe("LM Studio")
    expect(modelLocationsSummary([locations[2]!])).toBe("Magnitude isn’t looking anywhere else for models")
  })
  it("lists found automatic locations with every directory found and one on/off switch, and tucks away the ones not found", () => {
    expect(html).toContain("Hugging Face cache")
    expect(html).toMatch(/Hugging Face cache.*\/Volumes\/hf\/hub.*\/Users\/me\/.cache\/huggingface\/hub.*Look for models in Hugging Face cache/)
    expect(html).toContain("/Users/me/.lmstudio/models")
    expect(html).not.toContain("/Users/me/.cache/lm-studio/models")
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
  it("keeps directories it can't read in view with what to do, and summarizes only locations with one found", () => {
    const unreadable: ModelLocation[] = [
      locations[0]!,
      { kind: "Ollama", paths: [
        { path: "/home/me/.ollama/models", state: "Found" },
        { path: "/usr/share/ollama/.ollama/models", state: "Unreadable" },
        { path: "/var/lib/ollama", state: "NotFound" },
      ], enabled: true, automatic: true, state: "Found" },
      { kind: "Folder", paths: [{ path: "/srv/models", state: "Unreadable" }], enabled: true, automatic: false, state: "Unreadable" },
    ]
    const html = manager(unreadable)
    expect(modelLocationsSummary(unreadable)).toBe("Hugging Face cache, Ollama · 1 folder")
    expect(modelLocationsSummary([{ ...unreadable[1]!, paths: [{ path: "/var/lib/ollama", state: "Unreadable" }], state: "Unreadable" }]))
      .toBe("Magnitude isn’t looking anywhere else for models")
    expect(html).toContain('aria-label="Look for models in Ollama"')
    expect(html).toContain("/home/me/.ollama/models")
    expect(html).toContain("Can’t read · /usr/share/ollama/.ollama/models")
    expect(html).not.toContain("/var/lib/ollama")
    expect(html).toContain("Can’t read · /srv/models")
    expect(html).toContain("Give your account read access to it.")
    expect(html).not.toContain("Not found on this computer")
  })
  it("tells a system Ollama install to grant read access or set OLLAMA_MODELS, and only Ollama", () => {
    const ollama: ModelLocation = { kind: "Ollama", paths: [{ path: "/usr/share/ollama/.ollama/models", state: "Unreadable" }], enabled: true, automatic: true, state: "Unreadable" }
    const html = manager([ollama])
    expect(html).toMatch(/Can’t read · \/usr\/share\/ollama\/.ollama\/models<\/span><span[^>]*>Magnitude doesn’t have permission to read this folder\. A system Ollama service keeps its models in a folder that belongs to the ollama user: give your account read access to it, or set OLLAMA_MODELS to a folder you can read \(Magnitude also scans the default folders\)\.</)
    const folder = manager([{ kind: "Folder", paths: [{ path: "/srv/models", state: "Unreadable" }], enabled: true, automatic: false, state: "Unreadable" }])
    expect(folder).toContain("Give your account read access to it.")
    expect(folder).not.toContain("OLLAMA_MODELS")
  })
  it("explains a first folder before any exist", () => {
    expect(manager(locations.slice(0, 1))).toContain("Add a folder of GGUF files no other app manages")
    expect(manager(locations.slice(0, 1))).not.toContain("<details")
  })
  it("words folder and reference failures without service text", () => {
    expect(modelLocationErrorNotice(new DirectoryNotFound({ path: "/x" })).title).toBe("This folder doesn’t exist")
    expect(candidateErrorNotice(new InvalidModelReference({ reference: "x", message: "PRIVATE" }))).toMatchObject({ title: "Magnitude can’t open this model", retryable: false })
    expect(candidateErrorNotice(new ModelSourceUnavailable({ reason: "SourceUnreachable", message: "PRIVATE", retryable: true }))).toMatchObject({ title: "Couldn’t reach Hugging Face", retryable: true })
    expect(candidateErrorNotice(new ModelSourceUnavailable({ reason: "ServiceUnreachable", message: "PRIVATE", retryable: true })).title).toBe("Couldn’t reach the Magnitude service")
    expect(JSON.stringify(candidateErrorNotice(new ModelSourceUnavailable({ reason: "Unavailable", message: "PRIVATE", retryable: false })))).not.toContain("PRIVATE")
    const access = "This repository requires accepting its terms on Hugging Face and an access token. Set HF_TOKEN to a token from an account that has accepted them, then try again."
    expect(candidateErrorNotice(new ModelSourceUnavailable({ reason: "AccessRequired", message: access, retryable: false })))
      .toMatchObject({ title: "This repository needs access", description: access, retryable: false })
    const missing = "Repository not found, or it's private or gated. Check the name, or set HF_TOKEN if you have access."
    expect(candidateErrorNotice(new ModelRepositoryNotFound({ reference: "nobody-xyz/does-not-exist-GGUF", message: missing })))
      .toEqual({ title: "Magnitude can’t find this repository", description: missing, retryable: false })
    const unreadable = candidateErrorNotice(new ModelSourceUnavailable({ reason: "Unreadable", message: "Magnitude couldn't read this file.", retryable: false }))
    expect(unreadable).toMatchObject({ title: "Magnitude couldn’t read this model", description: "Magnitude couldn't read this file." })
    expect(unreadable.title).not.toMatch(/reach/i)
    expect(candidateErrorNotice(new Error("transport")).title).toBe("Couldn’t reach the Magnitude service")
  })
  it("words a refused drafter from the engine's reasons", () => {
    expect(candidateErrorNotice(new DrafterNotUsable({
      reference: "z-lab/drafter:dflash.gguf",
      reasons: [{ _tag: "TargetMismatch", detail: "PRIVATE" }],
    }))).toEqual({
      title: "This drafter can’t be used",
      description: "This drafter can't be used with this model: Its drafter wasn't built for this model.",
      retryable: false,
    })
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

describe("native open dialog", () => {
  const folder = { kind: "folder" as const, title: "Add a folder of models", defaultPath: Option.none<string>() }
  it("uses the native dialog whenever the window has a native host, on any OS, and the in-app browser otherwise", () => {
    const calls: string[] = []
    const native: NativeOpenPath = (request, onChosen) => { calls.push(`native ${request.kind}`); onChosen("C:\\Models") }
    choosePath(Option.some(native), folder, path => calls.push(`chose ${path}`), () => calls.push("browse"))()
    expect(calls).toEqual(["native folder", "chose C:\\Models"])
    calls.length = 0
    choosePath(Option.none(), folder, path => calls.push(`chose ${path}`), () => calls.push("browse"))()
    expect(calls).toEqual(["browse"])
  })
  it("adds nothing when the native dialog is cancelled", () => {
    const calls: string[] = []
    choosePath(Option.some(() => {}), folder, path => calls.push(path), () => calls.push("browse"))()
    expect(calls).toEqual([])
  })
})
