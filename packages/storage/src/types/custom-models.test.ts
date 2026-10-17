import { Either, Option, Schema } from 'effect'
import { describe, expect, it } from 'vitest'

import { MagnitudeConfigSchema } from './config'
import { CustomModelsConfigSchema, EMPTY_CUSTOM_MODELS_CONFIG } from './custom-models'

const digest = '0123456789abcdef'.repeat(4)
const decode = Schema.decodeUnknownEither(CustomModelsConfigSchema)

const config = {
  declarations: {
    'hf:HauhauCS/Qwen3.6-Uncensored-GGUF/Qwen3.6-Q4_K_P.gguf': {
      origin: { _tag: 'Added', repository: 'HauhauCS/Qwen3.6-Uncensored-GGUF', commit: '4f1c2e9' },
      target: ['Qwen3.6-Q4_K_P.gguf'],
      projector: 'mmproj-F16.gguf',
    },
    [`local:${digest}`]: {
      origin: { _tag: 'Imported', path: '/Users/me/models/model.gguf', headerDigest: digest },
      target: ['/Users/me/models/model.gguf'],
    },
  },
  hidden: ['hf:owner/repository/hidden.gguf'],
  drafters: {
    [`local:${digest}`]: { _tag: 'Reference', reference: { _tag: 'Catalog', modelId: 'qwen3.6-35b-a3b:gguf:q4' }, method: { _tag: 'DFlash' } },
    'hf:owner/repository/hidden.gguf': { _tag: 'None' },
  },
  folders: ['/Volumes/models'],
  disabledLocations: ['Ollama'],
}

describe('custom models configuration', () => {
  it('materializes an empty configuration', () => {
    expect(EMPTY_CUSTOM_MODELS_CONFIG).toEqual({
      declarations: {},
      hidden: [],
      drafters: {},
      folders: [],
      disabledLocations: [],
    })
  })

  it('decodes declarations, drafters, folders, and locations and round-trips them', () => {
    const decoded = Either.getOrThrowWith(decode(config), String)
    const added = decoded.declarations['hf:HauhauCS/Qwen3.6-Uncensored-GGUF/Qwen3.6-Q4_K_P.gguf' as keyof typeof decoded.declarations]
    expect(added?.projector).toEqual(Option.some('mmproj-F16.gguf'))
    const imported = decoded.declarations[`local:${digest}` as keyof typeof decoded.declarations]
    expect(imported?.projector).toEqual(Option.none())
    const encoded = Schema.encodeSync(CustomModelsConfigSchema)(decoded)
    expect(encoded).toEqual(config)
  })

  it('rejects a declaration whose origin disagrees with its model ID', () => {
    expect(Either.isLeft(decode({
      declarations: {
        'hf:owner/repository/model.gguf': {
          origin: { _tag: 'Added', repository: 'owner/other', commit: 'abc' },
          target: ['model.gguf'],
        },
      },
    }))).toBe(true)
    expect(Either.isLeft(decode({
      declarations: {
        [`local:${digest}`]: {
          origin: { _tag: 'Imported', path: '/model.gguf', headerDigest: 'f'.repeat(64) },
          target: ['/model.gguf'],
        },
      },
    }))).toBe(true)
  })

  it('drops declarations keyed by a catalog model ID', () => {
    const decoded = Either.getOrThrowWith(decode({
      declarations: { 'qwen3.5-4b:gguf:q4': { origin: { _tag: 'Imported', path: '/m.gguf', headerDigest: digest }, target: ['/m.gguf'] } },
    }), String)
    expect(decoded.declarations).toEqual({})
  })

  it('rejects empty targets and duplicate folders', () => {
    expect(Either.isLeft(decode({
      declarations: {
        'hf:owner/repository/model.gguf': {
          origin: { _tag: 'Added', repository: 'owner/repository', commit: 'abc' },
          target: [],
        },
      },
    }))).toBe(true)
    expect(Either.isLeft(decode({ folders: ['/a', '/a'] }))).toBe(true)
    expect(Either.isLeft(decode({ disabledLocations: ['Folder'] }))).toBe(true)
  })

  it('mounts under models in the Magnitude configuration', () => {
    const decoded = Schema.decodeUnknownSync(MagnitudeConfigSchema)({ models: { folders: ['/Volumes/models'] } })
    expect(Option.map(decoded.models, ({ folders }) => folders)).toEqual(Option.some(['/Volumes/models']))
    expect(Schema.decodeUnknownSync(MagnitudeConfigSchema)({}).models).toEqual(Option.none())
  })
})
