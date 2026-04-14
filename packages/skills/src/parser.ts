import { Data, Effect, ParseResult, Schema } from 'effect'
import { remark } from 'remark'
import remarkParse from 'remark-parse'
import remarkFrontmatter from 'remark-frontmatter'
import type { Root, Html } from 'mdast'
import { SkillSchema, ThinkingLensSchema, type Skill, type SkillSections, type ThinkingLens } from './types'

export class SkillParseError extends Data.TaggedError('SkillParseError')<{
  readonly cause: ParseResult.ParseError
}> {}

type SectionKey = 'shared' | 'lead' | 'worker' | 'handoff'

const MARKER_RE = /^<!--\s*@(shared|lead|worker|handoff)\s*-->$/

const processor = remark().use(remarkParse).use(remarkFrontmatter, ['yaml'])

const ThinkingArraySchema = Schema.Array(ThinkingLensSchema)

function parseFrontmatter(tree: Root) {
  const yamlNode = tree.children.find((n) => n.type === 'yaml')
  if (!yamlNode || !('value' in yamlNode)) {
    return Effect.succeed({ name: '', description: '', thinking: [] as readonly ThinkingLens[] })
  }

  const data = (Bun.YAML.parse(yamlNode.value as string) ?? {}) as Record<string, unknown>

  return Effect.gen(function* () {
    const name = yield* Schema.decodeUnknown(Schema.String)(data.name ?? '')
    const description = yield* Schema.decodeUnknown(Schema.String)(data.description ?? '')
    const thinking = Array.isArray(data.thinking)
      ? yield* Schema.decodeUnknown(ThinkingArraySchema)(data.thinking)
      : []

    return { name, description, thinking }
  })
}

function splitSections(body: string, tree: Root): SkillSections {
  const acc: Record<SectionKey, string[]> = { shared: [], lead: [], worker: [], handoff: [] }
  const markers: Array<{ key: SectionKey; start: number; end: number }> = []

  for (const node of tree.children) {
    if (node.type !== 'html') continue
    const m = (node as Html).value.trim().match(MARKER_RE)
    if (!m) continue
    const start = node.position?.start.offset
    const end = node.position?.end.offset
    if (typeof start !== 'number' || typeof end !== 'number') continue
    markers.push({ key: m[1] as SectionKey, start, end })
  }

  if (markers.length === 0) {
    const trimmed = body.trim()
    if (trimmed) acc.shared.push(trimmed)
  } else {
    const preamble = body.slice(0, markers[0].start).trim()
    if (preamble) acc.shared.push(preamble)

    for (let i = 0; i < markers.length; i++) {
      const chunk = body.slice(markers[i].end, markers[i + 1]?.start ?? body.length).trim()
      if (chunk) acc[markers[i].key].push(chunk)
    }
  }

  return {
    shared: acc.shared.join('\n\n'),
    lead: acc.lead.join('\n\n'),
    worker: acc.worker.join('\n\n'),
    handoff: acc.handoff.join('\n\n'),
  }
}

export function parseSkill(content: string): Effect.Effect<Skill, SkillParseError> {
  return Effect.gen(function* () {
    const tree = processor.parse(content) as Root
    const { name, description, thinking } = yield* parseFrontmatter(tree)

    const yamlNode = tree.children.find((n) => n.type === 'yaml')
    const bodyStart = yamlNode?.position?.end.offset ?? 0
    const body = content.slice(bodyStart)

    const sections = splitSections(body, processor.parse(body) as Root)

    return yield* Schema.decodeUnknown(SkillSchema)({
      name,
      description,
      thinking,
      sections,
    })
  }).pipe(Effect.mapError((cause) => new SkillParseError({ cause })))
}
