import { remark } from 'remark'
import remarkGfm from 'remark-gfm'
import wikiLinkPlugin from 'remark-wiki-link'
import type { Root, PhrasingContent, Node } from 'mdast'

// remark-wiki-link types are incomplete — augment with actual options and node type
declare module 'remark-wiki-link' {
  interface WikiLinkPluginOptions {
    permalinks?: string[]
    pageResolver?: (name: string) => string[]
    newClassName?: string
    wikiLinkClassName?: string
    hrefTemplate?: (permalink: string) => string
    aliasDivider?: string
  }
}

// Augment mdast with WikiLink node type produced by remark-wiki-link
declare module 'mdast' {
  interface WikiLink extends Node {
    type: 'wikiLink'
    value: string
    data: {
      alias: string
      permalink: string
      exists: boolean
      hName: string
      hProperties: { className: string; href: string }
      hChildren: Array<{ type: string; value: string }>
    }
  }
  interface PhrasingContentMap {
    wikiLink: WikiLink
  }
}

const parser = remark()
  .use(remarkGfm)
  .use(wikiLinkPlugin, { aliasDivider: '|' })

export function parseMarkdownToMdast(content: string): Root {
  const tree = parser.runSync(parser.parse(content)) as Root
  ;((tree as any).data ??= {}).source = content
  return tree
}