import { describe, expect, it } from "vitest"
import { join, resolve } from "node:path"
import { resolveHuggingFaceCacheRoots } from "./hugging-face-cache"

const existing = () => true

describe("Hugging Face cache discovery", () => {
  it("uses HF_HUB_CACHE before every fallback", () => {
    expect(resolveHuggingFaceCacheRoots({
      env: {
        HF_HUB_CACHE: "/custom/hub",
        HUGGINGFACE_HUB_CACHE: "/legacy/hub",
        HF_HOME: "/hf-home",
        XDG_CACHE_HOME: "/xdg",
      },
      homeDirectory: "/home/user",
      isDirectory: existing,
    })).toEqual(["/custom/hub"])
  })

  it("supports the legacy cache override when the current override is absent", () => {
    expect(resolveHuggingFaceCacheRoots({
      env: { HUGGINGFACE_HUB_CACHE: "/legacy/hub", HF_HOME: "/hf-home" },
      homeDirectory: "/home/user",
      isDirectory: existing,
    })).toEqual(["/legacy/hub"])
  })

  it("derives the hub from HF_HOME before XDG_CACHE_HOME", () => {
    expect(resolveHuggingFaceCacheRoots({
      env: { HF_HOME: "/hf-home", XDG_CACHE_HOME: "/xdg" },
      homeDirectory: "/home/user",
      isDirectory: existing,
    })).toEqual([join("/hf-home", "hub")])
  })

  it("uses XDG_CACHE_HOME when no Hugging Face override is configured", () => {
    expect(resolveHuggingFaceCacheRoots({
      env: { XDG_CACHE_HOME: "/xdg" },
      homeDirectory: "/home/user",
      isDirectory: existing,
    })).toEqual([join("/xdg", "huggingface", "hub")])
  })

  it("falls back to the user cache and ignores a missing directory", () => {
    const expected = join("/home/user", ".cache", "huggingface", "hub")
    expect(resolveHuggingFaceCacheRoots({
      env: {},
      homeDirectory: "/home/user",
      isDirectory: (path) => path === expected,
    })).toEqual([expected])
    expect(resolveHuggingFaceCacheRoots({
      env: {},
      homeDirectory: "/home/user",
      isDirectory: () => false,
    })).toEqual([])
  })

  it("expands home-relative overrides and normalizes relative overrides", () => {
    expect(resolveHuggingFaceCacheRoots({
      env: { HF_HUB_CACHE: "~/models/hub" },
      homeDirectory: "/home/user",
      isDirectory: existing,
    })).toEqual([join("/home/user", "models", "hub")])
    expect(resolveHuggingFaceCacheRoots({
      env: { HF_HUB_CACHE: "relative/hub" },
      homeDirectory: "/home/user",
      isDirectory: existing,
    })).toEqual([resolve("relative/hub")])
  })
})
