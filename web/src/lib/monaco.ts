import * as monaco from "monaco-editor"
import editorWorker from "monaco-editor/editor/editor.worker?worker"
import jsonWorker from "monaco-editor/language/json/json.worker?worker"
import cssWorker from "monaco-editor/language/css/css.worker?worker"
import htmlWorker from "monaco-editor/language/html/html.worker?worker"
import tsWorker from "monaco-editor/language/typescript/ts.worker?worker"
import { loader } from "@monaco-editor/react"

self.MonacoEnvironment = {
  getWorker(_moduleId: string, label: string) {
    if (label === "json") return new jsonWorker()
    if (label === "css" || label === "scss" || label === "less") return new cssWorker()
    if (label === "html" || label === "handlebars" || label === "razor") return new htmlWorker()
    if (label === "typescript" || label === "javascript") return new tsWorker()
    return new editorWorker()
  },
}

loader.config({ monaco })
monaco.editor.defineTheme("magnitude-dark", {
  base: "vs-dark",
  inherit: true,
  rules: [],
  colors: {
    "editor.background": "#172131",
    "editor.foreground": "#cbd5e1",
    "editorLineNumber.foreground": "#64748b",
    "editorLineNumber.activeForeground": "#cbd5e1",
    "editorCursor.foreground": "#38bdf8",
    "editor.selectionBackground": "#1e40af88",
    "editor.inactiveSelectionBackground": "#33415588",
    "editorIndentGuide.background1": "#293548",
    "editorIndentGuide.activeBackground1": "#475569",
  },
})
monaco.editor.defineTheme("magnitude-light", {
  base: "vs",
  inherit: true,
  rules: [],
  colors: {
    "editor.background": "#ffffff",
    "editor.foreground": "#1e293b",
    "editorLineNumber.foreground": "#94a3b8",
    "editorLineNumber.activeForeground": "#475569",
    "editorCursor.foreground": "#0369a1",
    "editor.selectionBackground": "#bae6fd",
    "editor.inactiveSelectionBackground": "#e2e8f0",
  },
})
export { monaco }
