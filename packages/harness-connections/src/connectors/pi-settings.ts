import { Effect, Option, Schema } from "effect"
import { parse, printParseErrorCode, type ParseError } from "jsonc-parser"
import { readOr, updateJsonc } from "../shared"
import { PiPackageSourceSchema } from "./pi-package-state"

export const PiPackageConfigurationSchema = Schema.Struct({
  source: PiPackageSourceSchema,
  autoload: Schema.optionalWith(Schema.Boolean, { as: "Option", exact: true }),
  extensions: Schema.optionalWith(Schema.Array(Schema.String), { as: "Option", exact: true }),
})
export const PiPackageEntrySchema = Schema.Union(PiPackageSourceSchema, PiPackageConfigurationSchema)
export type PiPackageEntry = typeof PiPackageEntrySchema.Type

const PackagesSchema = Schema.Array(PiPackageEntrySchema)
export const PiSettingsSchema = Schema.Struct({
  packages: Schema.optionalWith(PackagesSchema, { as: "Option", exact: true }),
  defaultProvider: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  defaultModel: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
})
export type PiSettings = typeof PiSettingsSchema.Type

export class PiSettingsInvalid extends Schema.TaggedError<PiSettingsInvalid>()("PiSettingsInvalid", {
  path: Schema.String,
  detail: Schema.String,
}) {
  override get message() { return `Invalid Pi settings at ${this.path}: ${this.detail}` }
}

export interface PiSettingsDocument {
  readonly text: string
  readonly settings: PiSettings
}

// Keep unknown host fields through decoding and recovery encoding. Ordinary edits
// patch the original JSONC text rather than replacing it with this typed view.
const decodeSettings = Schema.decodeUnknown(PiSettingsSchema, { onExcessProperty: "preserve" })
const encodePackages = Schema.encode(PackagesSchema, { onExcessProperty: "preserve" })

export const decodePiSettings = (text: string, path: string): Effect.Effect<PiSettingsDocument, PiSettingsInvalid> =>
  Effect.gen(function* () {
    const errors: ParseError[] = []
    const value: unknown = parse(text, errors, { allowTrailingComma: true, disallowComments: false })
    if (errors.length > 0) {
      return yield* new PiSettingsInvalid({
        path,
        detail: errors.map(({ error, offset }) => `${printParseErrorCode(error)} at offset ${offset}`).join("; "),
      })
    }
    const settings = yield* decodeSettings(value).pipe(
      Effect.mapError((error) => new PiSettingsInvalid({ path, detail: error.message })),
    )
    return { text, settings }
  })

export const readPiSettings = (path: string) => readOr(path, "{}\n").pipe(
  Effect.flatMap((text) => decodePiSettings(text, path)),
)

export const piPackageSource = (entry: PiPackageEntry) => typeof entry === "string" ? entry : entry.source
export const piPackageFilters = (entry: PiPackageEntry): Option.Option<readonly string[]> =>
  typeof entry === "string" ? Option.none() : entry.extensions

/** Used only for native-operation recovery; preserve each entry's unrelated fields. */
export const replacePiPackages = (document: PiSettingsDocument, packages: Option.Option<readonly PiPackageEntry[]>) =>
  Effect.gen(function* () {
    const encoded = yield* Option.match(packages, {
      onNone: () => Effect.void,
      onSome: encodePackages,
    })
    return updateJsonc(document.text, [[["packages"], encoded]])
  })
