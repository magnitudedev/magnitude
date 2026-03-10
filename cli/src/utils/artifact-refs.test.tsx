import { describe, expect, it } from 'bun:test'
import React, { type ReactNode } from 'react'
import { injectArtifactRefsWithHitZones } from './artifact-refs'

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

const refStyle = { fg: '#blue' }
const hoverStyle = { fg: '#hover' }
const notFoundStyle = { fg: '#gray' }
const isValid = (_name: string) => true

type Case = {
  name: string
  md: string
  expectedArtifacts: string[]
  expectedLabels?: string[]
}

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
  { name: 'Ref inside table cell', md: '| See [[my-ref]] | done |', expectedArtifacts: ['my-ref'] },
  { name: 'Ref with section', md: '[[my-ref#section]]', expectedArtifacts: ['my-ref'], expectedLabels: ['my-ref#section'] },
  { name: 'Ref with label', md: '[[my-ref|Custom Label]]', expectedArtifacts: ['my-ref'], expectedLabels: ['Custom Label'] },
  { name: 'Ref next to inline code', md: 'Use `code` then [[my-ref]]', expectedArtifacts: ['my-ref'] },
  { name: 'Ref inside link text', md: '[text [[my-ref]]](url)', expectedArtifacts: ['my-ref'] },
  {
    name: 'Multiple refs with styled text between',
    md: '**Bold** [[ref-1]] and *italic* [[ref-2]]',
    expectedArtifacts: ['ref-1', 'ref-2'],
  },
  {
    name: 'Ref at paragraph boundary (two paragraphs)',
    md: 'Para 1 [[ref-1]]\n\nPara 2 [[ref-2]]',
    expectedArtifacts: ['ref-1', 'ref-2'],
  },
  { name: 'No refs at all', md: 'Just plain text', expectedArtifacts: [] },
]

// Additional cases that go through the full markdown rendering pipeline
// (parseMarkdownToChunks → convertLinesToReactNodes → injectArtifactRefsWithHitZones)
import { parseMarkdownToChunks } from './markdown-content-renderer'
import { buildMarkdownColorPalette, chatThemes } from './theme'

const fullPipelineCases: Case[] = [
  { name: 'Plain paragraph with ref (full pipeline)', md: 'Check [[my-ref]] here', expectedArtifacts: ['my-ref'] },
  { name: 'Ref with section and space (full pipeline)', md: 'See [[demo-notes#Key Points]] for info', expectedArtifacts: ['demo-notes'], expectedLabels: ['demo-notes#Key Points'] },
  { name: 'Ref in blockquote with admonition (full pipeline)', md: '> [!NOTE]\n> You can reference [[demo-notes#Key Points]] for the important bits.', expectedArtifacts: ['demo-notes'], expectedLabels: ['demo-notes#Key Points'] },
  { name: 'Ref in blockquote (full pipeline)', md: '> Check [[sample-config]] for config.', expectedArtifacts: ['sample-config'] },
  { name: 'Multiple refs in paragraph (full pipeline)', md: 'Both [[demo-notes]] and [[sample-config]] are available.', expectedArtifacts: ['demo-notes', 'sample-config'] },
  { name: 'Ref in bold (full pipeline)', md: '**See [[my-ref]]**', expectedArtifacts: ['my-ref'] },
  { name: 'Ref in list item (full pipeline)', md: '- Check [[my-ref]] here', expectedArtifacts: ['my-ref'] },
  { name: 'Ref in table cell (full pipeline)', md: '| Feature | Notes |\n|---------|-------|\n| Artifacts | See [[demo-notes]] |', expectedArtifacts: ['demo-notes'] },
]

describe('injectArtifactRefsWithHitZones with full rendering pipeline', () => {
  const palette = buildMarkdownColorPalette(chatThemes.dark)

  for (const tc of fullPipelineCases) {
    it(tc.name, () => {
      const chunks = parseMarkdownToChunks(tc.md, { palette })
      const textChunks = chunks.filter(c => c.type === 'text')

      // Collect all hitZones across all text chunks
      const allHitZones: Array<{ artifactName: string; section?: string }> = []
      let allText = ''

      for (const chunk of textChunks) {
        const result = injectArtifactRefsWithHitZones(
          chunk.content,
          refStyle, hoverStyle, notFoundStyle, isValid,
        )
        allHitZones.push(...result.hitZones)
        allText += flattenToText(result.node)
      }

      if (tc.expectedArtifacts.length === 0) {
        expect(allHitZones).toHaveLength(0)
        return
      }

      expect(allHitZones).toHaveLength(tc.expectedArtifacts.length)
      tc.expectedArtifacts.forEach((artifactName, i) => {
        expect(allHitZones[i]?.artifactName).toBe(artifactName)
      })

      const expectedLabels = tc.expectedLabels ?? tc.expectedArtifacts
      expectedLabels.forEach((label) => {
        expect(allText).toContain(`[≡ ${label}]`)
      })
    })
  }
})

describe('injectArtifactRefsWithHitZones with real Bun markdown output', () => {
  for (const tc of cases) {
    it(tc.name, () => {
      const tree = Bun.markdown.react(tc.md)
      const result = injectArtifactRefsWithHitZones(tree, refStyle, hoverStyle, notFoundStyle, isValid)

      const text = flattenToText(result.node)

      if (tc.expectedArtifacts.length === 0) {
        expect(result.hitZones).toHaveLength(0)
        expect(text).toContain('Just plain text')
        expect(text).not.toContain('[≡ ')
        return
      }

      expect(result.hitZones).toHaveLength(tc.expectedArtifacts.length)

      tc.expectedArtifacts.forEach((artifactName, i) => {
        expect(result.hitZones[i]?.artifactName).toBe(artifactName)
      })

      const expectedLabels = tc.expectedLabels ?? tc.expectedArtifacts
      expectedLabels.forEach((label) => {
        expect(text).toContain(`[≡ ${label}]`)
      })
    })
  }
})