import { FileSystem } from "@effect/platform"
import { SystemError } from "@effect/platform/Error"
import { BunFileSystem } from "@effect/platform-bun"
import { Effect, Option } from "effect"
import { afterEach, describe, expect, it } from "vitest"
import { mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from "node:fs"
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
  it("lists every configured cache, then the default", () => {
    expect(huggingFaceCacheCandidates(linux({
      HF_HUB_CACHE: "/custom/hub", HUGGINGFACE_HUB_CACHE: "/legacy/hub", HF_HOME: "/hf", XDG_CACHE_HOME: "/xdg",
    }))).toEqual(["/custom/hub", "/legacy/hub", "/hf/hub", "/xdg/huggingface/hub", "/home/ada/.cache/huggingface/hub"])
    expect(huggingFaceCacheCandidates(linux({ HF_HUB_CACHE: "/custom/hub" })))
      .toEqual(["/custom/hub", "/home/ada/.cache/huggingface/hub"])
    expect(huggingFaceCacheCandidates(windows({ HF_HOME: "D:\\hf" })))
      .toEqual(["D:\\hf\\hub", "C:\\Users\\ada\\.cache\\huggingface\\hub"])
  })

  it("lists a directory named twice once and expands a tilde", () => {
    expect(huggingFaceCacheCandidates(mac())).toEqual(["/Users/ada/.cache/huggingface/hub"])
    expect(huggingFaceCacheCandidates(mac({ HF_HOME: "~/.cache/huggingface" }))).toEqual(["/Users/ada/.cache/huggingface/hub"])
    expect(huggingFaceCacheCandidates(mac({ HF_HUB_CACHE: "/hub", HUGGINGFACE_HUB_CACHE: "/hub" })))
      .toEqual(["/hub", "/Users/ada/.cache/huggingface/hub"])
    expect(huggingFaceCacheCandidates(windows())).toEqual(["C:\\Users\\ada\\.cache\\huggingface\\hub"])
  })

  it("ignores blank overrides", () => {
    expect(huggingFaceCacheCandidates(linux({ HF_HUB_CACHE: "  " }))).toEqual(["/home/ada/.cache/huggingface/hub"])
  })
})

describe("LM Studio location", () => {
  it("uses the current home and the older cache folder", () => {
    expect(lmStudioCandidates(mac(), noLmStudioSettings)).toEqual([
      "/Users/ada/.lmstudio/models",
      "/Users/ada/.cache/lm-studio/models",
    ])
    expect(lmStudioCandidates(windows(), noLmStudioSettings))
      .toEqual(["C:\\Users\\ada\\.lmstudio\\models", "C:\\Users\\ada\\.cache\\lm-studio\\models"])
  })

  it("follows a relocated home and a configured models folder first", () => {
    expect(lmStudioCandidates(linux(), {
      homePointer: Option.some("/data/lmstudio"),
      downloadsFolder: Option.some("/mnt/models"),
    })).toEqual(["/mnt/models", "/data/lmstudio/models", "/home/ada/.cache/lm-studio/models"])
    expect(lmStudioCandidates(linux(), { homePointer: Option.none(), downloadsFolder: Option.some("~/.lmstudio/models") }))
      .toEqual(["/home/ada/.lmstudio/models", "/home/ada/.cache/lm-studio/models"])
  })
})

describe("llama.cpp location", () => {
  it("lists LLAMA_CACHE before each platform's cache folder", () => {
    expect(llamaCppCandidates(windows({ LLAMA_CACHE: "D:\\llama" })))
      .toEqual(["D:\\llama", "C:\\Users\\ada\\AppData\\Local\\llama.cpp"])
    expect(llamaCppCandidates(mac({ LLAMA_CACHE: "/llama" }))).toEqual(["/llama", "/Users/ada/Library/Caches/llama.cpp"])
    expect(llamaCppCandidates(linux({ LLAMA_CACHE: "/llama", XDG_CACHE_HOME: "/xdg" })))
      .toEqual(["/llama", "/xdg/llama.cpp", "/home/ada/.cache/llama.cpp"])
  })

  it("uses each platform's cache folder", () => {
    expect(llamaCppCandidates(mac())).toEqual(["/Users/ada/Library/Caches/llama.cpp"])
    expect(llamaCppCandidates(linux())).toEqual(["/home/ada/.cache/llama.cpp"])
    expect(llamaCppCandidates(linux({ XDG_CACHE_HOME: "/xdg" }))).toEqual(["/xdg/llama.cpp", "/home/ada/.cache/llama.cpp"])
    expect(llamaCppCandidates(linux({ XDG_CACHE_HOME: "/home/ada/.cache" }))).toEqual(["/home/ada/.cache/llama.cpp"])
    expect(llamaCppCandidates(windows({ LOCALAPPDATA: "C:\\Users\\ada\\AppData\\Local" })))
      .toEqual(["C:\\Users\\ada\\AppData\\Local\\llama.cpp"])
    expect(llamaCppCandidates(windows())).toEqual(["C:\\Users\\ada\\AppData\\Local\\llama.cpp"])
  })
})

describe("Ollama location", () => {
  it("lists OLLAMA_MODELS before the user and system installs", () => {
    expect(ollamaCandidates(linux({ OLLAMA_MODELS: "/srv/ollama" })))
      .toEqual(["/srv/ollama", "/home/ada/.ollama/models", "/usr/share/ollama/.ollama/models", "/var/lib/ollama"])
    expect(ollamaCandidates(mac({ OLLAMA_MODELS: "/Volumes/ollama" }))).toEqual(["/Volumes/ollama", "/Users/ada/.ollama/models"])
    expect(ollamaCandidates(linux({ OLLAMA_MODELS: "/var/lib/ollama" })))
      .toEqual(["/var/lib/ollama", "/home/ada/.ollama/models", "/usr/share/ollama/.ollama/models"])
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

  it("reports every candidate with its state and reads LM Studio's own settings", async () => {
    const home = mkdtempSync(join(tmpdir(), "magnitude-discovery-roots-"))
    roots.push(home)
    const relocated = join(home, "lm-models")
    mkdirSync(relocated, { recursive: true })
    mkdirSync(join(home, ".lmstudio", "models"), { recursive: true })
    writeFileSync(join(home, ".lmstudio", "settings.json"), JSON.stringify({ downloadsFolder: relocated }))
    mkdirSync(join(home, ".cache", "huggingface", "hub"), { recursive: true })
    const resolved = await Effect.runPromise(resolveAutomaticLocations({
      platform: "darwin", env: { OLLAMA_MODELS: join(home, "ollama"), HF_HUB_CACHE: join(home, "hub") }, home,
    }).pipe(Effect.provide(BunFileSystem.layer)))
    expect(resolved).toEqual([
      { kind: "HuggingFaceCache", state: "Found", paths: [
        { path: join(home, "hub"), state: "NotFound" },
        { path: join(home, ".cache", "huggingface", "hub"), state: "Found" },
      ] },
      { kind: "LmStudio", state: "Found", paths: [
        { path: relocated, state: "Found" },
        { path: join(home, ".lmstudio", "models"), state: "Found" },
        { path: join(home, ".cache", "lm-studio", "models"), state: "NotFound" },
      ] },
      { kind: "LlamaCpp", state: "NotFound", paths: [{ path: join(home, "Library", "Caches", "llama.cpp"), state: "NotFound" }] },
      { kind: "Ollama", state: "NotFound", paths: [
        { path: join(home, "ollama"), state: "NotFound" },
        { path: join(home, ".ollama", "models"), state: "NotFound" },
      ] },
    ])
  })

  it("lists a candidate that links to another candidate once", async () => {
    const home = mkdtempSync(join(tmpdir(), "magnitude-discovery-roots-"))
    roots.push(home)
    mkdirSync(join(home, ".ollama", "models"), { recursive: true })
    symlinkSync(join(home, ".ollama", "models"), join(home, "ollama-link"))
    const resolved = await Effect.runPromise(resolveAutomaticLocations({
      platform: "darwin", env: { OLLAMA_MODELS: join(home, "ollama-link") }, home,
    }).pipe(Effect.provide(BunFileSystem.layer)))
    expect(resolved.find(({ kind }) => kind === "Ollama"))
      .toEqual({ kind: "Ollama", state: "Found", paths: [{ path: join(home, "ollama-link"), state: "Found" }] })
  })
})

describe("unreadable locations", () => {
  const denied = (method: string, path: string) =>
    new SystemError({ reason: "PermissionDenied", module: "FileSystem", method, pathOrDescriptor: path })
  const missing = (method: string, path: string) =>
    new SystemError({ reason: "NotFound", module: "FileSystem", method, pathOrDescriptor: path })
  const directory = { type: "Directory" } as FileSystem.File.Info
  const host = (
    directories: readonly string[],
    options: { readonly untraversable?: string; readonly unlistable?: string; readonly links?: Readonly<Record<string, string>> } = {},
  ) =>
    FileSystem.layerNoop({
      stat: (path) => options.untraversable !== undefined && path.startsWith(`${options.untraversable}/`)
        ? Effect.fail(denied("stat", path))
        : directories.includes(path) || path === options.unlistable ? Effect.succeed(directory) : Effect.fail(missing("stat", path)),
      readDirectory: (path) => path === options.unlistable ? Effect.fail(denied("readDirectory", path)) : Effect.succeed([]),
      readFileString: (path) => Effect.fail(missing("readFileString", path)),
      realPath: (path) => directories.includes(path) ? Effect.succeed(options.links?.[path] ?? path) : Effect.fail(missing("realPath", path)),
    })
  const ollama = (layer: ReturnType<typeof host>) => Effect.runPromise(resolveAutomaticLocations(linux()).pipe(
    Effect.map((locations) => locations.find(({ kind }) => kind === "Ollama")),
    Effect.provide(layer),
  ))

  it("reports a system service folder behind a private home as unreadable rather than missing", async () => {
    expect(await ollama(host([], { untraversable: "/usr/share/ollama" }))).toEqual({ kind: "Ollama", state: "Unreadable", paths: [
      { path: "/home/ada/.ollama/models", state: "NotFound" },
      { path: "/usr/share/ollama/.ollama/models", state: "Unreadable" },
      { path: "/var/lib/ollama", state: "NotFound" },
    ] })
  })

  it("is found when any directory is found, reporting each directory's own state", async () => {
    expect(await ollama(host(["/home/ada/.ollama/models", "/var/lib/ollama"], { untraversable: "/usr/share/ollama" })))
      .toEqual({ kind: "Ollama", state: "Found", paths: [
        { path: "/home/ada/.ollama/models", state: "Found" },
        { path: "/usr/share/ollama/.ollama/models", state: "Unreadable" },
        { path: "/var/lib/ollama", state: "Found" },
      ] })
  })

  it("keeps the first spelling of a system folder linked to another candidate", async () => {
    expect(await ollama(host(["/usr/share/ollama/.ollama/models", "/var/lib/ollama"], {
      links: { "/usr/share/ollama/.ollama/models": "/var/lib/ollama" },
    }))).toEqual({ kind: "Ollama", state: "Found", paths: [
      { path: "/home/ada/.ollama/models", state: "NotFound" },
      { path: "/usr/share/ollama/.ollama/models", state: "Found" },
    ] })
  })

  it("is not found when no candidate exists", async () => {
    expect((await ollama(host([])))?.state).toBe("NotFound")
  })

  it("reports a directory it can stat but not list as unreadable", async () => {
    const state = (path: string) => Effect.runPromise(directoryState(path).pipe(
      Effect.provide(host(["/srv/open"], { unlistable: "/srv/locked" }))))
    expect(await state("/srv/open")).toBe("Found")
    expect(await state("/srv/locked")).toBe("Unreadable")
    expect(await state("/srv/none")).toBe("NotFound")
  })
})
