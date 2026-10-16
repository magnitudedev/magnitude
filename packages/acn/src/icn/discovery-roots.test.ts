import { FileSystem } from "@effect/platform"
import { SystemError } from "@effect/platform/Error"
import { BunFileSystem } from "@effect/platform-bun"
import { Effect, Option } from "effect"
import { afterEach, describe, expect, it } from "vitest"
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import {
  huggingFaceCacheCandidates,
  llamaCppCandidates,
  lmStudioCandidates,
  directoryState,
  ollamaCandidates,
  resolveAutomaticLocations,
  type DiscoveryEnvironment,
} from "./discovery-roots"

const mac = (env: DiscoveryEnvironment["env"] = {}): DiscoveryEnvironment => ({ platform: "darwin", env, home: "/Users/ada" })
const linux = (env: DiscoveryEnvironment["env"] = {}): DiscoveryEnvironment => ({ platform: "linux", env, home: "/home/ada" })
const windows = (env: DiscoveryEnvironment["env"] = {}): DiscoveryEnvironment => ({ platform: "win32", env, home: "C:\\Users\\ada" })
const noLmStudioSettings = { homePointer: Option.none<string>(), downloadsFolder: Option.none<string>() }

describe("Hugging Face cache location", () => {
  it("prefers HF_HUB_CACHE, then the legacy override, HF_HOME, and XDG_CACHE_HOME", () => {
    expect(huggingFaceCacheCandidates(linux({
      HF_HUB_CACHE: "/custom/hub", HUGGINGFACE_HUB_CACHE: "/legacy/hub", HF_HOME: "/hf", XDG_CACHE_HOME: "/xdg",
    }))).toEqual(["/custom/hub"])
    expect(huggingFaceCacheCandidates(linux({ HUGGINGFACE_HUB_CACHE: "/legacy/hub", HF_HOME: "/hf" }))).toEqual(["/legacy/hub"])
    expect(huggingFaceCacheCandidates(linux({ HF_HOME: "/hf", XDG_CACHE_HOME: "/xdg" }))).toEqual(["/hf/hub"])
    expect(huggingFaceCacheCandidates(linux({ XDG_CACHE_HOME: "/xdg" }))).toEqual(["/xdg/huggingface/hub"])
  })

  it("falls back to the home cache and expands a tilde", () => {
    expect(huggingFaceCacheCandidates(mac())).toEqual(["/Users/ada/.cache/huggingface/hub"])
    expect(huggingFaceCacheCandidates(mac({ HF_HOME: "~/hf" }))).toEqual(["/Users/ada/hf/hub"])
    expect(huggingFaceCacheCandidates(windows())).toEqual(["C:\\Users\\ada\\.cache\\huggingface\\hub"])
  })

  it("ignores blank overrides", () => {
    expect(huggingFaceCacheCandidates(linux({ HF_HUB_CACHE: "  " }))).toEqual(["/home/ada/.cache/huggingface/hub"])
  })
})

describe("LM Studio location", () => {
  it("uses the current home, then the older cache folder", () => {
    expect(lmStudioCandidates(mac(), noLmStudioSettings)).toEqual([
      "/Users/ada/.lmstudio/models",
      "/Users/ada/.cache/lm-studio/models",
    ])
    expect(lmStudioCandidates(windows(), noLmStudioSettings)[0]).toBe("C:\\Users\\ada\\.lmstudio\\models")
  })

  it("follows a relocated home and a configured models folder first", () => {
    expect(lmStudioCandidates(linux(), {
      homePointer: Option.some("/data/lmstudio"),
      downloadsFolder: Option.some("/mnt/models"),
    })).toEqual(["/mnt/models", "/data/lmstudio/models", "/home/ada/.cache/lm-studio/models"])
  })
})

describe("llama.cpp location", () => {
  it("prefers LLAMA_CACHE on every platform", () => {
    expect(llamaCppCandidates(windows({ LLAMA_CACHE: "D:\\llama" }))).toEqual(["D:\\llama"])
    expect(llamaCppCandidates(mac({ LLAMA_CACHE: "/llama" }))).toEqual(["/llama"])
  })

  it("uses each platform's cache folder", () => {
    expect(llamaCppCandidates(mac())).toEqual(["/Users/ada/Library/Caches/llama.cpp"])
    expect(llamaCppCandidates(linux())).toEqual(["/home/ada/.cache/llama.cpp"])
    expect(llamaCppCandidates(linux({ XDG_CACHE_HOME: "/xdg" }))).toEqual(["/xdg/llama.cpp"])
    expect(llamaCppCandidates(windows({ LOCALAPPDATA: "C:\\Users\\ada\\AppData\\Local" })))
      .toEqual(["C:\\Users\\ada\\AppData\\Local\\llama.cpp"])
    expect(llamaCppCandidates(windows())).toEqual(["C:\\Users\\ada\\AppData\\Local\\llama.cpp"])
  })
})

describe("Ollama location", () => {
  it("prefers OLLAMA_MODELS", () => {
    expect(ollamaCandidates(linux({ OLLAMA_MODELS: "/srv/ollama" }))).toEqual(["/srv/ollama"])
  })

  it("adds the Linux service installs after the user folder", () => {
    expect(ollamaCandidates(mac())).toEqual(["/Users/ada/.ollama/models"])
    expect(ollamaCandidates(linux())).toEqual(["/home/ada/.ollama/models", "/usr/share/ollama/.ollama/models", "/var/lib/ollama"])
    expect(ollamaCandidates(windows())).toEqual(["C:\\Users\\ada\\.ollama\\models"])
  })
})

describe("automatic location resolution", () => {
  const roots: string[] = []
  afterEach(() => { roots.splice(0).forEach((root) => rmSync(root, { recursive: true, force: true })) })

  it("picks the first existing candidate and reads LM Studio's own settings", async () => {
    const home = mkdtempSync(join(tmpdir(), "magnitude-discovery-roots-"))
    roots.push(home)
    const relocated = join(home, "lm-models")
    mkdirSync(relocated, { recursive: true })
    mkdirSync(join(home, ".lmstudio"), { recursive: true })
    writeFileSync(join(home, ".lmstudio", "settings.json"), JSON.stringify({ downloadsFolder: relocated }))
    mkdirSync(join(home, ".cache", "huggingface", "hub"), { recursive: true })
    const resolved = await Effect.runPromise(resolveAutomaticLocations({
      platform: "linux", env: { OLLAMA_MODELS: join(home, "ollama") }, home,
    }).pipe(Effect.provide(BunFileSystem.layer)))
    expect(resolved).toEqual([
      { kind: "HuggingFaceCache", path: join(home, ".cache", "huggingface", "hub"), state: "Found" },
      { kind: "LmStudio", path: relocated, state: "Found" },
      { kind: "LlamaCpp", path: join(home, ".cache", "llama.cpp"), state: "NotFound" },
      { kind: "Ollama", path: join(home, "ollama"), state: "NotFound" },
    ])
  })
})

describe("unreadable locations", () => {
  const denied = (method: string, path: string) =>
    new SystemError({ reason: "PermissionDenied", module: "FileSystem", method, pathOrDescriptor: path })
  const missing = (method: string, path: string) =>
    new SystemError({ reason: "NotFound", module: "FileSystem", method, pathOrDescriptor: path })
  const directory = { type: "Directory" } as FileSystem.File.Info
  const host = (directories: readonly string[], options: { readonly untraversable?: string; readonly unlistable?: string } = {}) =>
    FileSystem.layerNoop({
      stat: (path) => options.untraversable !== undefined && path.startsWith(`${options.untraversable}/`)
        ? Effect.fail(denied("stat", path))
        : directories.includes(path) || path === options.unlistable ? Effect.succeed(directory) : Effect.fail(missing("stat", path)),
      readDirectory: (path) => path === options.unlistable ? Effect.fail(denied("readDirectory", path)) : Effect.succeed([]),
      readFileString: (path) => Effect.fail(missing("readFileString", path)),
    })
  const ollama = (layer: ReturnType<typeof host>) => Effect.runPromise(resolveAutomaticLocations(linux()).pipe(
    Effect.map((locations) => locations.find(({ kind }) => kind === "Ollama")),
    Effect.provide(layer),
  ))

  it("reports a system service folder behind a private home as unreadable rather than missing", async () => {
    expect(await ollama(host([], { untraversable: "/usr/share/ollama" })))
      .toEqual({ kind: "Ollama", path: "/usr/share/ollama/.ollama/models", state: "Unreadable" })
  })

  it("prefers a readable candidate over an earlier unreadable one", async () => {
    expect(await ollama(host(["/var/lib/ollama"], { untraversable: "/usr/share/ollama" })))
      .toEqual({ kind: "Ollama", path: "/var/lib/ollama", state: "Found" })
    expect(await ollama(host(["/home/ada/.ollama/models", "/var/lib/ollama"])))
      .toEqual({ kind: "Ollama", path: "/home/ada/.ollama/models", state: "Found" })
  })

  it("falls back to the first candidate when none exists", async () => {
    expect(await ollama(host([]))).toEqual({ kind: "Ollama", path: "/home/ada/.ollama/models", state: "NotFound" })
  })

  it("reports a directory it can stat but not list as unreadable", async () => {
    const state = (path: string) => Effect.runPromise(directoryState(path).pipe(
      Effect.provide(host(["/srv/open"], { unlistable: "/srv/locked" }))))
    expect(await state("/srv/open")).toBe("Found")
    expect(await state("/srv/locked")).toBe("Unreadable")
    expect(await state("/srv/none")).toBe("NotFound")
  })
})
