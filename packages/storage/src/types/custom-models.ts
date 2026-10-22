import {
  CatalogFormModelIdSchema,
  CustomFormModelIdSchema,
  HuggingFaceArtifactSelectorSchema,
  HuggingFaceFormModelIdSchema,
  HuggingFaceRepositoryIdSchema,
  ModelIdSchema,
} from '@magnitudedev/acn-protocol'
import { Option, ParseResult, Schema } from 'effect'

const PathSchema = Schema.NonEmptyTrimmedString.pipe(Schema.maxLength(4_096))
const AbsolutePathSchema = PathSchema.pipe(Schema.filter((path) => path.startsWith('/') || /^[A-Za-z]:[\\/]/.test(path), {
  message: () => 'path must be absolute',
}))

const isRecord = (value: unknown): value is Readonly<Record<string, unknown>> =>
  typeof value === 'object' && value !== null && !Array.isArray(value)

/** A record of which only the valid entries are kept: an invalid entry is dropped on its own, never the record. */
const validEntries = <K extends string, KI extends string, V, VI>(
  key: Schema.Schema<K, KI>,
  value: Schema.Schema<V, VI>,
) => Schema.transform(
  Schema.Unknown,
  Schema.partialWith(Schema.Record({ key: Schema.typeSchema(key), value: Schema.typeSchema(value) }), { exact: true }),
  {
    strict: false,
    decode: (input) => Object.fromEntries(Object.entries(isRecord(input) ? input : {}).flatMap(([entryKey, entryValue]) => {
      const decodedKey = Schema.decodeUnknownOption(key)(entryKey)
      const decodedValue = Schema.decodeUnknownOption(value)(entryValue)
      return Option.isSome(decodedKey) && Option.isSome(decodedValue) ? [[decodedKey.value, decodedValue.value]] : []
    })),
    encode: (entries) => Object.fromEntries(Object.entries(entries).map(([entryKey, entryValue]) =>
      [Schema.encodeSync(key)(entryKey as K), Schema.encodeSync(value)(entryValue as V)])),
  },
)

/** A list of which only the valid, distinct items are kept: an invalid item is dropped on its own, never the list. */
const validItems = <A extends string>(item: Schema.Schema<A, string>) => Schema.transform(
  Schema.Unknown,
  Schema.Array(Schema.typeSchema(item)),
  {
    strict: false,
    decode: (input) => [...new Set((Array.isArray(input) ? input : []).flatMap((entry) =>
      Option.toArray(Schema.decodeUnknownOption(item)(entry))))],
    encode: (items) => items.map((entry) => Schema.encodeSync(item)(entry)),
  },
)

/** The automatic scan locations, by the name `config.json` gives each. */
export const ModelLocationNameSchema = Schema.Literal('huggingface', 'lmstudio', 'llamacpp', 'ollama')
export type ModelLocationName = typeof ModelLocationNameSchema.Type

/** A model added from Hugging Face: the commit it is pinned to, and its projector's path in the repository. */
export const AddedModelSettingSchema = Schema.Struct({
  revision: Schema.NonEmptyTrimmedString,
  projector: Schema.optionalWith(PathSchema, { as: 'Option', exact: true }),
})
export type AddedModelSetting = typeof AddedModelSettingSchema.Type

/**
 * The drafter a custom model is served with, as `config.json` spells it: `"none"`, `"builtin"` (its own MTP heads),
 * a catalog model ID (that model's drafter), `"hf:<owner>/<repo>/<path>@<revision>"` (a drafter on the Hub pinned to a
 * revision), or an absolute path (a drafter on disk). A model without one uses automatic choice.
 */
export const DrafterSettingValueSchema = Schema.Union(
  Schema.TaggedStruct('None', {}),
  Schema.TaggedStruct('BuiltIn', {}),
  Schema.TaggedStruct('Catalog', { modelId: CatalogFormModelIdSchema }),
  Schema.TaggedStruct('HuggingFace', {
    repository: HuggingFaceRepositoryIdSchema,
    path: HuggingFaceArtifactSelectorSchema,
    revision: Schema.NonEmptyTrimmedString,
  }),
  Schema.TaggedStruct('Local', { path: AbsolutePathSchema }),
)
export type DrafterSetting = typeof DrafterSettingValueSchema.Type

const decodeDrafter = (value: string): Option.Option<unknown> => {
  if (value === 'none') return Option.some({ _tag: 'None' })
  if (value === 'builtin') return Option.some({ _tag: 'BuiltIn' })
  if (value.startsWith('hf:')) {
    const at = value.lastIndexOf('@')
    const parts = value.slice(3, at < 0 ? undefined : at).split('/')
    return at < 0 ? Option.none() : Option.some({
      _tag: 'HuggingFace',
      repository: parts.slice(0, 2).join('/'),
      path: parts.slice(2).join('/'),
      revision: value.slice(at + 1),
    })
  }
  if (Schema.is(AbsolutePathSchema)(value)) return Option.some({ _tag: 'Local', path: value })
  return Option.some({ _tag: 'Catalog', modelId: value })
}

const encodeDrafter = (setting: DrafterSetting): string => {
  switch (setting._tag) {
    case 'None': return 'none'
    case 'BuiltIn': return 'builtin'
    case 'Catalog': return setting.modelId
    case 'HuggingFace': return `hf:${setting.repository}/${setting.path}@${setting.revision}`
    case 'Local': return setting.path
  }
}

export const DrafterSettingSchema = Schema.transformOrFail(
  Schema.String,
  Schema.typeSchema(DrafterSettingValueSchema),
  {
    strict: true,
    decode: (value, _, ast) => Option.match(decodeDrafter(value), {
      onNone: () => ParseResult.fail(new ParseResult.Type(ast, value, 'not a drafter')),
      onSome: (setting) => ParseResult.decodeUnknown(DrafterSettingValueSchema)(setting).pipe(
        ParseResult.mapError(() => new ParseResult.Type(ast, value, 'not a drafter')),
      ),
    }),
    encode: (setting) => ParseResult.succeed(encodeDrafter(setting)),
  },
)

/**
 * Every user choice about custom models, under `models` in `config.json`, which users edit: automatic locations turned
 * on or off (`locations`; a missing one is on), folders and single GGUF files to scan (`paths`), models added from
 * Hugging Face (`added`), found models removed from the list (`hidden`), and drafters (`drafters`). It holds only
 * intent, in plain strings and readable IDs; its own schema, never the wire protocol's, so a protocol change never
 * changes the file. Each entry is validated on its own.
 */
export const CustomModelsConfigSchema = Schema.Struct({
  locations: Schema.optionalWith(validEntries(ModelLocationNameSchema, Schema.Boolean), { default: () => ({}) }),
  paths: Schema.optionalWith(validItems(AbsolutePathSchema), { default: () => [] }),
  added: Schema.optionalWith(validEntries(HuggingFaceFormModelIdSchema, AddedModelSettingSchema), { default: () => ({}) }),
  hidden: Schema.optionalWith(validItems(ModelIdSchema), { default: () => [] }),
  drafters: Schema.optionalWith(validEntries(CustomFormModelIdSchema, DrafterSettingSchema), { default: () => ({}) }),
})
export type CustomModelsConfig = Schema.Schema.Type<typeof CustomModelsConfigSchema>

export const EMPTY_CUSTOM_MODELS_CONFIG: CustomModelsConfig = Schema.decodeUnknownSync(CustomModelsConfigSchema)({})
