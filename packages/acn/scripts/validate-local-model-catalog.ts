import { LOCAL_MODEL_CATALOG } from "../src/local-inference/catalog"

interface HuggingFaceTreeFile {
  readonly type: "file" | "directory"
  readonly path: string
  readonly size: number
  readonly lfs?: { readonly oid: string; readonly size: number }
}

const groups = new Map<string, typeof LOCAL_MODEL_CATALOG[number][]>()
for (const entry of LOCAL_MODEL_CATALOG) {
  const key = `${entry.repo}@${entry.revision}`
  groups.set(key, [...(groups.get(key) ?? []), entry])
}

const failures: string[] = []
for (const [key, entries] of groups) {
  const entry = entries[0]!
  const url = `https://huggingface.co/api/models/${entry.repo}/tree/${entry.revision}?recursive=true&limit=1000`
  const response = await fetch(url)
  if (!response.ok) {
    failures.push(`${key}: Hugging Face returned ${response.status}`)
    continue
  }
  const tree = await response.json() as HuggingFaceTreeFile[]
  const files = new Map(tree.filter((item) => item.type === "file").map((item) => [item.path, item]))
  for (const catalogEntry of entries) {
    for (const expected of catalogEntry.files) {
      const actual = files.get(expected.path)
      if (!actual) {
        failures.push(`${catalogEntry.id}: missing ${expected.path}`)
        continue
      }
      if (actual.size !== expected.sizeBytes) {
        failures.push(`${catalogEntry.id}: ${expected.path} size ${actual.size} != ${expected.sizeBytes}`)
      }
      if (actual.lfs?.oid !== expected.sha256) {
        failures.push(`${catalogEntry.id}: ${expected.path} SHA-256 ${actual.lfs?.oid ?? "missing"} != ${expected.sha256}`)
      }
    }
  }
}

if (failures.length > 0) {
  for (const failure of failures) console.error(`ERROR ${failure}`)
  process.exit(1)
}

console.log(`Validated ${LOCAL_MODEL_CATALOG.length} pinned local-model artifacts across ${groups.size} Hugging Face repositories.`)

