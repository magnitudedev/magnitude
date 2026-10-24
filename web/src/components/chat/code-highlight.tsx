import type { ReactNode } from "react"
import { Effect } from "effect"
import { createLowlight } from "lowlight"
import bash from "highlight.js/lib/languages/bash"
import c from "highlight.js/lib/languages/c"
import cpp from "highlight.js/lib/languages/cpp"
import csharp from "highlight.js/lib/languages/csharp"
import css from "highlight.js/lib/languages/css"
import diff from "highlight.js/lib/languages/diff"
import dockerfile from "highlight.js/lib/languages/dockerfile"
import go from "highlight.js/lib/languages/go"
import ini from "highlight.js/lib/languages/ini"
import java from "highlight.js/lib/languages/java"
import javascript from "highlight.js/lib/languages/javascript"
import json from "highlight.js/lib/languages/json"
import markdown from "highlight.js/lib/languages/markdown"
import python from "highlight.js/lib/languages/python"
import rust from "highlight.js/lib/languages/rust"
import sql from "highlight.js/lib/languages/sql"
import typescript from "highlight.js/lib/languages/typescript"
import xml from "highlight.js/lib/languages/xml"
import yaml from "highlight.js/lib/languages/yaml"

const highlighter = createLowlight({
  bash,
  c,
  cpp,
  csharp,
  css,
  diff,
  dockerfile,
  go,
  ini,
  java,
  javascript,
  json,
  markdown,
  python,
  rust,
  sql,
  typescript,
  xml,
  yaml,
})
highlighter.registerAlias({
  bash: ["shell"],
  javascript: ["jsx"],
  typescript: ["tsx"],
  ini: ["toml"],
})
const labels: Readonly<Record<string, string>> = {
  bash: "Bash",
  sh: "Shell",
  shell: "Shell",
  zsh: "Shell",
  c: "C",
  cpp: "C++",
  "c++": "C++",
  csharp: "C#",
  cs: "C#",
  css: "CSS",
  diff: "Diff",
  dockerfile: "Dockerfile",
  docker: "Dockerfile",
  go: "Go",
  golang: "Go",
  ini: "INI",
  toml: "TOML",
  java: "Java",
  javascript: "JavaScript",
  js: "JavaScript",
  jsx: "JSX",
  json: "JSON",
  markdown: "Markdown",
  md: "Markdown",
  python: "Python",
  py: "Python",
  rust: "Rust",
  rs: "Rust",
  sql: "SQL",
  typescript: "TypeScript",
  ts: "TypeScript",
  tsx: "TSX",
  xml: "XML",
  html: "HTML",
  yaml: "YAML",
  yml: "YAML",
  text: "Text",
  txt: "Text",
  plaintext: "Text",
}
export const codeLanguageLabel = (language: string) =>
  labels[language.toLowerCase()] ?? (language || "Text")

type SyntaxNode = ReturnType<typeof highlighter.highlight>["children"][number]
const syntaxNodes = (nodes: readonly SyntaxNode[]): ReactNode =>
  nodes.map((node, i) =>
    node.type === "text" ? (
      node.value
    ) : node.type === "element" ? (
      <span
        key={i}
        className={
          Array.isArray(node.properties.className)
            ? node.properties.className.join(" ")
            : undefined
        }
      >
        {syntaxNodes(node.children)}
      </span>
    ) : null
  )

/** No guessing languages or HTML injection; large/unknown blocks remain readable text. */
export const highlightCode = (code: string, language: string): ReactNode => {
  const normalized = language.toLowerCase()
  if (code.length > 50_000 || !highlighter.registered(normalized)) return code
  return Effect.try(() =>
    syntaxNodes(highlighter.highlight(normalized, code).children)
  ).pipe(
    Effect.orElseSucceed(() => code),
    Effect.runSync
  )
}
