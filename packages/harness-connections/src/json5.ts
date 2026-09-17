import { Schema } from "effect"
import { parse, update } from "jju"

const ObjectSchema = Schema.Record({ key: Schema.String, value: Schema.Unknown })

export const json5Object = (source: string): Record<string, unknown> =>
  Schema.decodeUnknownSync(ObjectSchema)(parse(source, { mode: "json5", reserved_keys: "throw" }))

/** Edit only the requested fields while retaining JSON5 comments and formatting. */
export const updateJson5 = (
  source: string,
  changes: ReadonlyArray<readonly [ReadonlyArray<string>, unknown]>,
): string => {
  const document = json5Object(source)
  changes: for (const [segments, value] of changes) {
    let parent = document
    for (const key of segments.slice(0, -1)) {
      if (parent[key] === undefined) {
        if (value === undefined) continue changes
        parent[key] = {}
      }
      const child = parent[key]
      if (!Schema.is(ObjectSchema)(child)) throw new Error(`Configuration field ${key} is not an object`)
      parent = child
    }
    const key = segments.at(-1)
    if (key === undefined) throw new Error("Configuration field path cannot be empty")
    if (value === undefined) delete parent[key]
    else parent[key] = value
  }
  return update(source, document, { mode: "json5", reserved_keys: "throw" })
}
