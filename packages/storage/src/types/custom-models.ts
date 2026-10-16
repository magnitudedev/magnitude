import {
  AutomaticModelLocationKindSchema,
  CustomFormModelIdSchema,
  DrafterChoiceSchema,
  HuggingFaceRepositoryIdSchema,
  ModelHeaderDigestSchema,
  parseModelId,
  type CustomFormModelId,
} from '@magnitudedev/acn-protocol'
import { Schema } from 'effect'

const PathSchema = Schema.NonEmptyTrimmedString.pipe(Schema.maxLength(4_096))

export const CustomModelDeclarationOriginSchema = Schema.Union(
  Schema.TaggedStruct('Added', {
    repository: HuggingFaceRepositoryIdSchema,
    commit: Schema.NonEmptyTrimmedString,
  }),
  Schema.TaggedStruct('Imported', {
    path: PathSchema,
    headerDigest: ModelHeaderDigestSchema,
  }),
)
export type CustomModelDeclarationOrigin = Schema.Schema.Type<typeof CustomModelDeclarationOriginSchema>

/** An added or imported model: its pinned source, the target files (shards included), and its projector. */
export const CustomModelDeclarationSchema = Schema.Struct({
  origin: CustomModelDeclarationOriginSchema,
  target: Schema.NonEmptyArray(PathSchema),
  projector: Schema.optionalWith(PathSchema, { as: 'Option', exact: true }),
})
export type CustomModelDeclaration = Schema.Schema.Type<typeof CustomModelDeclarationSchema>

const declarationMatchesId = (modelId: CustomFormModelId, declaration: CustomModelDeclaration): boolean => {
  const identity = parseModelId(modelId)
  switch (declaration.origin._tag) {
    case 'Added':
      return identity._tag === 'HuggingFace'
        && identity.repositoryId === declaration.origin.repository
        && identity.artifactSelector === declaration.target[0]
    case 'Imported':
      return identity._tag === 'Local' && identity.headerDigest === declaration.origin.headerDigest
  }
}

const unique = <A>(values: ReadonlyArray<A>): boolean => new Set(values).size === values.length

/**
 * Every user choice about custom models, persisted by ACN under `models` in `config.json`: declarations,
 * hidden models, attached drafters, user folders, and turned-off automatic locations.
 */
export const CustomModelsConfigSchema = Schema.Struct({
  declarations: Schema.optionalWith(
    Schema.Record({ key: CustomFormModelIdSchema, value: CustomModelDeclarationSchema }),
    { default: () => ({}) },
  ),
  hidden: Schema.optionalWith(Schema.Array(CustomFormModelIdSchema), { default: () => [] }),
  drafters: Schema.optionalWith(
    Schema.Record({ key: CustomFormModelIdSchema, value: DrafterChoiceSchema }),
    { default: () => ({}) },
  ),
  folders: Schema.optionalWith(Schema.Array(PathSchema), { default: () => [] }),
  disabledLocations: Schema.optionalWith(Schema.Array(AutomaticModelLocationKindSchema), { default: () => [] }),
}).pipe(Schema.filter((config) =>
  Object.entries(config.declarations).every(([modelId, declaration]) =>
    declarationMatchesId(modelId as CustomFormModelId, declaration))
  && unique(config.hidden)
  && unique(config.folders)
  && unique(config.disabledLocations), {
  message: () => 'custom model declarations must match their IDs, and hidden models, folders, and disabled locations must be unique',
}))
export type CustomModelsConfig = Schema.Schema.Type<typeof CustomModelsConfigSchema>

export const EMPTY_CUSTOM_MODELS_CONFIG: CustomModelsConfig = Schema.decodeUnknownSync(CustomModelsConfigSchema)({})
