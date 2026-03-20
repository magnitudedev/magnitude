import { createLowlight, common } from 'lowlight'
import type { Element, RootContent as HastRootContent, Text as HastText } from 'hast'
import type { SyntaxColors } from './theme'
import type { Span } from './blocks'

const lowlight = createLowlight(common)

const extensionToLanguage: Record<string, string> = {
  ts: 'typescript',
  js: 'javascript',
  mjs: 'javascript',
  cjs: 'javascript',
  jsx: 'javascript',
  tsx: 'typescript',
  py: 'python',
  rb: 'ruby',
  rs: 'rust',
  sh: 'bash',
  zsh: 'bash',
  bash: 'bash',
  yml: 'yaml',
  yaml: 'yaml',
  json: 'json',
  md: 'markdown',
  css: 'css',
  html: 'xml',
  xml: 'xml',
  java: 'java',
  go: 'go',
  php: 'php',
  c: 'c',
  h: 'c',
  cpp: 'cpp',
  cc: 'cpp',
  cxx: 'cpp',
  hpp: 'cpp',
  cs: 'csharp',
  swift: 'swift',
  kt: 'kotlin',
  sql: 'sql',
  toml: 'ini',
  ini: 'ini',
}

export function inferLanguageFromFilename(filename: string): string | null {
  const name = filename.split('/').pop() ?? filename
  const ext = name.includes('.') ? name.split('.').pop()?.toLowerCase() : undefined
  if (!ext) return null
  const mapped = extensionToLanguage[ext] ?? ext
  return lowlight.registered(mapped) ? mapped : null
}

export function hljsClassToColor(classNames: string[], syntax: SyntaxColors): string {
  for (const cls of classNames) {
    switch (cls) {
      case 'hljs-keyword':
        return syntax.keyword
      case 'hljs-string':
      case 'hljs-template-string':
      case 'hljs-regexp':
        return syntax.string
      case 'hljs-number':
        return syntax.number
      case 'hljs-comment':
        return syntax.comment
      case 'hljs-title':
      case 'hljs-function':
        return syntax.function
      case 'hljs-variable':
      case 'hljs-attr':
      case 'hljs-params':
        return syntax.variable
      case 'hljs-type':
      case 'hljs-built_in':
      case 'hljs-class':
        return syntax.type
      case 'hljs-operator':
        return syntax.operator
      case 'hljs-property':
        return syntax.property
      case 'hljs-punctuation':
        return syntax.punctuation
      case 'hljs-literal':
        return syntax.literal
    }
  }
  return syntax.default
}

export function highlightToLines(nodes: HastRootContent[], syntax: SyntaxColors): Span[][] {
  const lines: Span[][] = [[]]

  const walk = (node: HastRootContent, inheritedColor?: string): void => {
    if (node.type === 'text') {
      const textNode = node as HastText
      const parts = textNode.value.split('\n')
      parts.forEach((part, idx) => {
        if (idx > 0) lines.push([])
        if (part) lines[lines.length - 1].push({ text: part, fg: inheritedColor ?? syntax.default })
      })
      return
    }

    if (node.type === 'element') {
      const el = node as Element
      const classNames = (el.properties?.className as string[]) ?? []
      const color = hljsClassToColor(classNames, syntax)
      for (const child of el.children) walk(child as HastRootContent, color)
    }
  }

  for (const node of nodes) walk(node)
  return lines
}

export function tryHighlight(code: string, lang: string, syntax: SyntaxColors): Span[][] | null {
  if (!lowlight.registered(lang)) return null
  try {
    const result = lowlight.highlight(lang, code)
    return highlightToLines(result.children as HastRootContent[], syntax)
  } catch {
    return null
  }
}

const defaultSyntaxColors: SyntaxColors = {
  keyword: '#a78bfa',
  string: '#86efac',
  number: '#93c5fd',
  comment: '#64748b',
  function: '#60a5fa',
  variable: '#e2e8f0',
  type: '#86efac',
  operator: '#94a3b8',
  property: '#e2e8f0',
  punctuation: '#64748b',
  literal: '#93c5fd',
  default: '#f1f5f9',
}

function plainTextLines(content: string): Span[][] {
  const lines = content.split('\n')
  return lines.map((line) => [{ text: line || ' ' }])
}

export function highlightFile(content: string, filename: string): Span[][] {
  const language = inferLanguageFromFilename(filename)
  if (!language) return plainTextLines(content)
  const highlighted = tryHighlight(content, language, defaultSyntaxColors)
  if (!highlighted) return plainTextLines(content)
  const rawLines = content.split('\n')
  return rawLines.map((lineText, idx) => highlighted[idx] ?? [{ text: lineText || ' ' }])
}
