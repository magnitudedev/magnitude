/**
 * Shared Shiki highlighter store — used by useSyncExternalStore.
 *
 * Module-level singleton: the highlighter is created once, cached,
 * and all components that need syntax highlighting share it.
 */
import type { Highlighter } from "shiki"
import { buildMergedPalette } from "@magnitudedev/client-common"

let highlighterPromise: Promise<Highlighter> | null = null
let highlighterValue: Highlighter | null = null
const listeners = new Set<() => void>()
const markdownPalette = buildMergedPalette()
const shikiTheme = {
  name: "magnitude-dark",
  type: "dark" as const,
  fg: markdownPalette.codeTextFg,
  bg: markdownPalette.codeBackground,
  colors: {
    "editor.background": markdownPalette.codeBackground,
    "editor.foreground": markdownPalette.codeTextFg,
  },
  settings: [
    { settings: { foreground: markdownPalette.syntax.default, background: markdownPalette.codeBackground } },
    { scope: ["comment", "punctuation.definition.comment"], settings: { foreground: markdownPalette.syntax.comment } },
    { scope: ["string", "constant.other.symbol"], settings: { foreground: markdownPalette.syntax.string } },
    { scope: ["constant.numeric", "constant.language"], settings: { foreground: markdownPalette.syntax.number } },
    { scope: ["keyword", "storage", "storage.type"], settings: { foreground: markdownPalette.syntax.keyword } },
    { scope: ["entity.name.function", "support.function"], settings: { foreground: markdownPalette.syntax.function } },
    { scope: ["entity.name.type", "support.type", "entity.name.class"], settings: { foreground: markdownPalette.syntax.type } },
    { scope: ["variable", "entity.name.variable"], settings: { foreground: markdownPalette.syntax.variable } },
    { scope: ["variable.other.property", "support.variable.property"], settings: { foreground: markdownPalette.syntax.property } },
    { scope: ["keyword.operator", "punctuation"], settings: { foreground: markdownPalette.syntax.operator } },
  ],
}

function ensureHighlighter(): void {
  if (highlighterPromise || highlighterValue) return
  highlighterPromise = import("shiki").then((shiki) =>
    shiki.createHighlighter({
      themes: [shikiTheme],
      langs: [
        "typescript", "tsx", "javascript", "jsx", "json", "bash", "shell",
        "python", "rust", "go", "css", "html", "markdown", "yaml", "sql",
        "diff", "toml", "xml", "java", "c", "cpp", "ruby", "text",
      ],
    }),
  )
  highlighterPromise.then((hl) => {
    highlighterValue = hl
    listeners.forEach((cb) => cb())
  }).catch(() => {
    highlighterPromise = null
  })
}

export function subscribeShiki(cb: () => void): () => void {
  listeners.add(cb)
  ensureHighlighter()
  return () => { listeners.delete(cb) }
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
      theme: "magnitude-dark",
    })
  } catch {
    return `<pre><code>${escapeHtml(code)}</code></pre>`
  }
}
