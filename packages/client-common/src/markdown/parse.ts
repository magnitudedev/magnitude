import { remark } from 'remark'
import remarkGfm from 'remark-gfm'
import type { Root } from 'mdast'

const parser = remark()
  .use(remarkGfm)

export function parseMarkdownToMdast(content: string): Root {
  const tree = parser.runSync(parser.parse(content)) as Root
  ;((tree as any).data ??= {}).source = content
  return tree
}
