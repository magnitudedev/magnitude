import { Either, Option, Schema } from 'effect'
import { describe, expect, it } from 'vitest'

import { MagnitudeConfigSchema } from './config'
import { CustomModelsConfigSchema, EMPTY_CUSTOM_MODELS_CONFIG } from './custom-models'

const decode = Schema.decodeUnknownEither(CustomModelsConfigSchema)
const added = 'hf:unsloth/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf'
const local = 'local:gemma-4-12b-instruct-q8_0:41ad09c2e7f3'
const finetune = 'local:my-finetune-q4_k_m:9f2c1a7b3e04'

const config = {
  locations: { ollama: false },
  paths: ['/Users/x/my-models', '/Users/x/finetune.gguf'],
  added: { [added]: { revision: 'abc123', projector: 'mmproj-F16.gguf' } },
  hidden: [local],
  drafters: {
    [added]: 'qwen3-8b:gguf:q4',
    [finetune]: 'none',
    [local]: 'hf:z-lab/Qwen3-8B-DFlash/model.gguf@def456',
    'hf:owner/repository/model.gguf': '/Users/x/drafter.gguf',
    'hf:owner/repository/other.gguf': 'builtin',
  },
}

describe('custom models configuration', () => {
  it('materializes an empty configuration', () => {
    expect(EMPTY_CUSTOM_MODELS_CONFIG).toEqual({ locations: {}, paths: [], added: {}, hidden: [], drafters: {} })
  })

  it('decodes the plain strings users write into choices and round-trips them', () => {
    const decoded = Either.getOrThrowWith(decode(config), String)
    expect(decoded.locations).toEqual({ ollama: false })
    expect(decoded.added[added as keyof typeof decoded.added]).toEqual({
      revision: 'abc123',
      projector: Option.some('mmproj-F16.gguf'),
    })
    expect(Object.values(decoded.drafters)).toEqual([
      { _tag: 'Catalog', modelId: 'qwen3-8b:gguf:q4' },
      { _tag: 'None' },
      { _tag: 'HuggingFace', repository: 'z-lab/Qwen3-8B-DFlash', path: 'model.gguf', revision: 'def456' },
      { _tag: 'Local', path: '/Users/x/drafter.gguf' },
      { _tag: 'BuiltIn' },
    ])
    expect(Schema.encodeSync(CustomModelsConfigSchema)(decoded)).toEqual(config)
  })

  it('drops each invalid entry on its own and keeps the rest', () => {
    const decoded = Either.getOrThrowWith(decode({
      locations: { ollama: false, folder: false, lmstudio: 'off' },
      paths: ['/models', 'relative/models', '/models', 7],
      added: {
        [added]: { revision: 'abc123' },
        'qwen3.5-4b:gguf:q4': { revision: 'abc123' },
        [local]: { revision: 'abc123' },
        'hf:owner/repository/model.gguf': { projector: 'mmproj.gguf' },
      },
      hidden: [local, 'local:41ad09c2e7f3', 'not a model'],
      drafters: {
        [finetune]: 'none',
        'qwen3.5-4b:gguf:q4': 'none',
        [local]: 'hf:z-lab/Qwen3-8B-DFlash/model.gguf',
        [added]: 'auto',
      },
    }), String)
    expect(decoded).toEqual({
      locations: { ollama: false },
      paths: ['/models'],
      added: { [added]: { revision: 'abc123', projector: Option.none() } },
      hidden: [local],
      drafters: { [finetune]: { _tag: 'None' } },
    })
  })

  it('keeps every other entry when a whole field has the wrong shape', () => {
    const decoded = Either.getOrThrowWith(decode({ paths: '/models', hidden: [local] }), String)
    expect(decoded.paths).toEqual([])
    expect(decoded.hidden).toEqual([local])
  })

  it('mounts under models in the Magnitude configuration', () => {
    const decoded = Schema.decodeUnknownSync(MagnitudeConfigSchema)({ models: { paths: ['/Volumes/models'] } })
    expect(Option.map(decoded.models, ({ paths }) => paths)).toEqual(Option.some(['/Volumes/models']))
    expect(Schema.decodeUnknownSync(MagnitudeConfigSchema)({}).models).toEqual(Option.none())
  })
})
