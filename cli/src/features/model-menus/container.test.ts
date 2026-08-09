import { describe, expect, test } from "vitest"
import type { KeyEvent } from "@opentui/core"
import {
  huggingFaceRepositoryUrls,
  resolveRootNavigationDirection,
  scrollCatalogCandidateIntoView,
} from "./container"
import { makeCatalogCandidate } from "../local-inference/test-fixtures"

const key = (
  name: string,
  overrides: Partial<Pick<KeyEvent, "ctrl" | "meta" | "option" | "shift">> = {},
): Pick<KeyEvent, "name" | "ctrl" | "meta" | "option" | "shift"> => ({
  name,
  ctrl: false,
  meta: false,
  option: false,
  shift: false,
  ...overrides,
})

describe("model menu root navigation", () => {
  test("resolves lateral navigation without depending on the nested view", () => {
    expect(resolveRootNavigationDirection(key("left"))).toBe(-1)
    expect(resolveRootNavigationDirection(key("right"))).toBe(1)
  })

  test("resolves forward and reverse tab navigation", () => {
    expect(resolveRootNavigationDirection(key("tab"))).toBe(1)
    expect(resolveRootNavigationDirection(key("tab", { shift: true }))).toBe(-1)
  })

  test("leaves modified navigation keys unhandled", () => {
    expect(resolveRootNavigationDirection(key("left", { ctrl: true }))).toBeNull()
    expect(resolveRootNavigationDirection(key("right", { meta: true }))).toBeNull()
    expect(resolveRootNavigationDirection(key("tab", { option: true }))).toBeNull()
  })

  test("leaves unrelated keys to the active view", () => {
    expect(resolveRootNavigationDirection(key("escape"))).toBeNull()
    expect(resolveRootNavigationDirection(key("up"))).toBeNull()
  })
})

describe("catalog keyboard navigation", () => {
  test("reveals the candidate selected by the keyboard cursor", () => {
    const revealed: string[] = []

    scrollCatalogCandidateIntoView({
      scrollChildIntoView: (id) => { revealed.push(id) },
    }, "qwen-config")

    expect(revealed).toEqual(["catalog-candidate:qwen-config"])
  })

  test("does nothing before the catalog scrollbox is mounted", () => {
    expect(() => scrollCatalogCandidateIntoView(null, "qwen-config")).not.toThrow()
  })
})

describe("catalog repository links", () => {
  test("derives unique Hugging Face repository URLs from package sources", () => {
    const candidate = makeCatalogCandidate({
      sources: [
        {
          source: {
            _tag: "HuggingFace",
            repository: "LiquidAI/LFM2.5-2.6B-GGUF",
            revision: "revision",
          },
          files: [],
        },
        {
          source: {
            _tag: "HuggingFace",
            repository: "LiquidAI/LFM2.5-2.6B-GGUF",
            revision: "revision",
          },
          files: [],
        },
        {
          source: { _tag: "Local", path: "/models/liquid" },
          files: [],
        },
      ],
    })

    expect(huggingFaceRepositoryUrls(candidate)).toEqual([
      "https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF",
    ])
  })

  test("returns no repository URL for local-only packages", () => {
    const candidate = makeCatalogCandidate({
      sources: [{
        source: { _tag: "Local", path: "/models/local-only" },
        files: [],
      }],
    })

    expect(huggingFaceRepositoryUrls(candidate)).toEqual([])
  })
})
