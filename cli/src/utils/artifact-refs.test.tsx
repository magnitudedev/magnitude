import { describe, expect, it } from 'bun:test'
import React, { type ReactNode } from 'react'
import { parseMarkdownToChunks, convertLinesToReactNodes } from './markdown-content-renderer'
import { buildMarkdownColorPalette, chatThemes } from './theme'

function flattenToText(node: ReactNode): string {
  if (node === null || node === undefined || typeof node === 'boolean') return ''
  if (typeof node === 'string' || typeof node === 'number') return String(node)
  if (Array.isArray(node)) return node.map(flattenToText).join('')
  if (React.isValidElement(node)) {
    const el = node as React.ReactElement<{ children?: ReactNode }>
    return flattenToText(el.props.children)
  }
  return ''
}

/** Check if a React tree contains elements with data-artifact-ref matching the expected name */
function findArtifactRefs(node: ReactNode): Array<{ artifactName: string; section?: string; label?: string }> {
  const refs: Array<{ artifactName: string; section?: string; label?: string }> = []

  function walk(n: ReactNode): void {
    if (n === null || n === undefined || typeof n === 'string' || typeof n === 'number' || typeof n === 'boolean') return
    if (Array.isArray(n)) { n.forEach(walk); return }
    if (React.isValidElement(n)) {
      const el = n as React.ReactElement<Record<string, unknown>>
      const ref = el.props['data-artifact-ref'] as string | undefined
      if (ref) {
        refs.push({
          artifactName: ref,
          section: el.props['data-artifact-section'] as string | undefined,
          label: el.props['data-artifact-label'] as string | undefined,
        })
      }
      if (el.props.children) walk(el.props.children as ReactNode)
    }
  }

  walk(node)
  return refs
}

type Case = {
  name: string
  md: string
  expectedArtifacts: string[]
  expectedSections?: (string | undefined)[]
}

const palette = buildMarkdownColorPalette(chatThemes.dark)

const cases: Case[] = [
  { name: 'Plain paragraph with ref', md: 'Check [[my-ref]] here', expectedArtifacts: ['my-ref'] },
  { name: 'Ref at start of text', md: '[[my-ref]] is great', expectedArtifacts: ['my-ref'] },
  { name: 'Ref at end of text', md: 'See [[my-ref]]', expectedArtifacts: ['my-ref'] },
  { name: 'Multiple refs', md: 'See [[ref-1]] and [[ref-2]]', expectedArtifacts: ['ref-1', 'ref-2'] },
  { name: 'Ref inside bold', md: '**See [[my-ref]]**', expectedArtifacts: ['my-ref'] },
  { name: 'Ref inside italic', md: '*See [[my-ref]]*', expectedArtifacts: ['my-ref'] },
  { name: 'Ref inside heading', md: '## Check [[my-ref]]', expectedArtifacts: ['my-ref'] },
  { name: 'Ref inside blockquote', md: '> Check [[my-ref]]', expectedArtifacts: ['my-ref'] },
  { name: 'Ref inside blockquote with bold', md: '> **Note:** Check [[my-ref]]', expectedArtifacts: ['my-ref'] },
  { name: 'Ref inside list item', md: '- See [[my-ref]]', expectedArtifacts: ['my-ref'] },
  { name: 'Ref inside ordered list', md: '1. See [[my-ref]]', expectedArtifacts: ['my-ref'] },
  { name: 'Ref inside table cell', md: '| See [[my-ref]] | done |\n|---|---|\n| a | b |', expectedArtifacts: ['my-ref'] },
  { name: 'Ref with section', md: '[[my-ref#section]]', expectedArtifacts: ['my-ref'], expectedSections: ['section'] },
  { name: 'Ref with section and space', md: 'See [[demo-notes#Key Points]] here', expectedArtifacts: ['demo-notes'], expectedSections: ['Key Points'] },
  { name: 'Ref next to inline code', md: 'Use `code` then [[my-ref]]', expectedArtifacts: ['my-ref'] },
  { name: 'Multiple refs with styled text between', md: '**Bold** [[ref-1]] and *italic* [[ref-2]]', expectedArtifacts: ['ref-1', 'ref-2'] },
  { name: 'Ref at paragraph boundary', md: 'Para 1 [[ref-1]]\n\nPara 2 [[ref-2]]', expectedArtifacts: ['ref-1', 'ref-2'] },
  { name: 'Ref in blockquote with admonition', md: '> [!NOTE]\n> You can reference [[demo-notes#Key Points]] for the important bits.', expectedArtifacts: ['demo-notes'], expectedSections: ['Key Points'] },
  { name: 'No refs at all', md: 'Just plain text', expectedArtifacts: [] },
]

describe('Artifact refs in markdown rendering pipeline', () => {
  for (const tc of cases) {
    it(tc.name, () => {
      const chunks = parseMarkdownToChunks(tc.md, { palette })
      const textChunks = chunks.filter(c => c.type === 'text')

      // Collect all refs across all text chunks
      const allRefs: Array<{ artifactName: string; section?: string }> = []
      for (const chunk of textChunks) {
        allRefs.push(...findArtifactRefs(convertLinesToReactNodes(chunk.lines)))
      }

      expect(allRefs).toHaveLength(tc.expectedArtifacts.length)
      tc.expectedArtifacts.forEach((name, i) => {
        expect(allRefs[i]?.artifactName).toBe(name)
      })
      if (tc.expectedSections) {
        tc.expectedSections.forEach((section, i) => {
          expect(allRefs[i]?.section).toBe(section)
        })
      }

      // Verify the original ref text [[...]] is preserved in the content
      // (it gets replaced with display label in StyledTextWithRefs, not here)
      if (tc.expectedArtifacts.length > 0) {
        const allText = textChunks.map(c => flattenToText(convertLinesToReactNodes(c.lines))).join('')
        for (const name of tc.expectedArtifacts) {
          expect(allText).toContain(name)
        }
      }
    })
  }
})