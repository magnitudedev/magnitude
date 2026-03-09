export interface ArtifactRef {
  readonly id: string
  readonly section?: string
}

const ARTIFACT_LINK_RE = /\[\[([^\]\n]+)\]\]/g

export function extractArtifactRefs(text: string): ArtifactRef[] {
  const refs: ArtifactRef[] = []
  const seen = new Set<string>()

  for (const match of text.matchAll(ARTIFACT_LINK_RE)) {
    const raw = match[1]?.trim()
    if (!raw) continue

    const hashIdx = raw.indexOf('#')
    const id = (hashIdx >= 0 ? raw.slice(0, hashIdx) : raw).trim()
    if (!id) continue
    const section = hashIdx >= 0 ? raw.slice(hashIdx + 1).trim() : ''

    const key = `${id}#${section}`
    if (seen.has(key)) continue
    seen.add(key)

    refs.push(section ? { id, section } : { id })
  }

  return refs
}