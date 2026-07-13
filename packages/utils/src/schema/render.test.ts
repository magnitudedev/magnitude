import { describe, expect, test } from 'vitest'
import { Schema } from 'effect'
import { renderSchemaParams, renderSchemaType } from './render'
import { inspectSchemaShape } from './shape'

describe('schema render utilities', () => {
  test('inspects recursive schemas as finite shape indexes', () => {
    type Tree = readonly Tree[]
    const TreeSchema: Schema.Schema<Tree> = Schema.suspend((): Schema.Schema<Tree> =>
      Schema.Array(TreeSchema) as Schema.Schema<Tree>
    )

    const shape = inspectSchemaShape(TreeSchema)
    const root = shape.get(shape.root)

    expect(root.kind).toBe('alias')
    if (root.kind !== 'alias') return
    expect(root.reason).toBe('suspend')
    expect(shape.get(root.target).kind).toBe('array')
  })

  test('renders record index signatures instead of dropping them', () => {
    const rendered = renderSchemaParams(Schema.Struct({
      bag: Schema.Record({ key: Schema.String, value: Schema.Number }),
    }))

    expect(rendered).toContain('bag: { [key: string]: number }')
  })

  test('renders anonymous recursive schemas without overflowing', () => {
    type Tree = readonly Tree[]
    const TreeSchema: Schema.Schema<Tree> = Schema.suspend((): Schema.Schema<Tree> =>
      Schema.Array(TreeSchema) as Schema.Schema<Tree>
    )

    expect(renderSchemaType(TreeSchema)).toContain('<recursive>')
  })
})
