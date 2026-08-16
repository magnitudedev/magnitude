/**
 * Shared Shiki highlighter store — used by useSyncExternalStore.
 *
 * Module-level singleton: the highlighter is created once, cached,
 * and all components that need syntax highlighting share it.
 */
import type { Highlighter } from "shiki"
import {
  blue,
  buildMergedPalette,
  green,
  slate,
  violet,
} from "@magnitudedev/client-common"
import { getResolvedAppearance, subscribeAppearance } from "./appearance-store"

let highlighterPromise: Promise<Highlighter> | null = null
let highlighterValue: Highlighter | null = null
const listeners = new Set<() => void>()
const markdownPalette = buildMergedPalette()
const darkTheme = {
  name: "magnitude-dark",
  type: "dark" as const,
  fg: markdownPalette.codeTextFg,
  bg: markdownPalette.codeBackground,
  colors: {
    "editor.background": markdownPalette.codeBackground,
    "editor.foreground": markdownPalette.codeTextFg,
  },
  settings: [
    {
      settings: {
        foreground: markdownPalette.syntax.default,
        background: markdownPalette.codeBackground,
      },
    },
    {
      scope: ["comment", "punctuation.definition.comment"],
      settings: { foreground: markdownPalette.syntax.comment },
    },
    {
      scope: ["string", "constant.other.symbol"],
      settings: { foreground: markdownPalette.syntax.string },
    },
    {
      scope: ["constant.numeric", "constant.language"],
      settings: { foreground: markdownPalette.syntax.number },
    },
    {
      scope: ["keyword", "storage", "storage.type"],
      settings: { foreground: markdownPalette.syntax.keyword },
    },
    {
      scope: ["entity.name.function", "support.function"],
      settings: { foreground: markdownPalette.syntax.function },
    },
    {
      scope: ["entity.name.type", "support.type", "entity.name.class"],
      settings: { foreground: markdownPalette.syntax.type },
    },
    {
      scope: ["variable", "entity.name.variable"],
      settings: { foreground: markdownPalette.syntax.variable },
    },
    {
      scope: ["variable.other.property", "support.variable.property"],
      settings: { foreground: markdownPalette.syntax.property },
    },
    {
      scope: ["keyword.operator", "punctuation"],
      settings: { foreground: markdownPalette.syntax.operator },
    },
  ],
}
const lightTheme = {
  name: "magnitude-light",
  type: "light" as const,
  fg: slate[900],
  bg: slate[100],
  colors: {
    "editor.background": slate[100],
    "editor.foreground": slate[900],
  },
  settings: [
    { settings: { foreground: slate[900], background: slate[100] } },
    {
      scope: ["comment", "punctuation.definition.comment"],
      settings: { foreground: slate[500] },
    },
    {
      scope: ["string", "constant.other.symbol"],
      settings: { foreground: green[700] },
    },
    {
      scope: ["constant.numeric", "constant.language"],
      settings: { foreground: blue[700] },
    },
    {
      scope: ["keyword", "storage", "storage.type"],
      settings: { foreground: violet[700] },
    },
    {
      scope: ["entity.name.function", "support.function"],
      settings: { foreground: blue[700] },
    },
    {
      scope: ["entity.name.type", "support.type", "entity.name.class"],
      settings: { foreground: green[700] },
    },
    {
      scope: ["variable", "entity.name.variable"],
      settings: { foreground: slate[700] },
    },
    {
      scope: ["variable.other.property", "support.variable.property"],
      settings: { foreground: blue[700] },
    },
    {
      scope: ["keyword.operator", "punctuation"],
      settings: { foreground: slate[700] },
    },
  ],
}

function ensureHighlighter(): void {
  if (highlighterPromise || highlighterValue) return
  highlighterPromise = import("shiki").then((shiki) =>
    shiki.createHighlighter({
      themes: [darkTheme, lightTheme],
      langs: [
        "typescript",
        "tsx",
        "javascript",
        "jsx",
        "json",
        "bash",
        "shell",
        "python",
        "rust",
        "go",
        "css",
        "html",
        "markdown",
        "yaml",
        "sql",
        "diff",
        "toml",
        "xml",
        "java",
        "c",
        "cpp",
        "ruby",
        "text",
      ],
    })
  )
  highlighterPromise
    .then((hl) => {
      highlighterValue = hl
      listeners.forEach((cb) => cb())
    })
    .catch(() => {
      highlighterPromise = null
    })
}

export function subscribeShiki(cb: () => void): () => void {
  listeners.add(cb)
  ensureHighlighter()
  const unsubscribeAppearance = subscribeAppearance(cb)
  return () => {
    listeners.delete(cb)
    unsubscribeAppearance()
  }
}

export function getShikiSnapshot(): Highlighter | null {
  return highlighterValue
}

function escapeHtml(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
}

/** Highlight code — returns HTML string, or null if highlighter not yet loaded */
export function highlightCode(code: string, lang: string): string | null {
  if (!highlighterValue) return null
  try {
    return highlighterValue.codeToHtml(code, {
      lang: lang || "text",
      theme:
        getResolvedAppearance() === "light"
          ? "magnitude-light"
          : "magnitude-dark",
    })
  } catch {
    return `<pre><code>${escapeHtml(code)}</code></pre>`
  }
}
