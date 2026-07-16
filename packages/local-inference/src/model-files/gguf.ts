import * as FileSystem from "@effect/platform/FileSystem"
import { Data, Effect, Option, pipe } from "effect"
import { ModelArtifactKey, ModelFileFormatId, type SourceFileKey } from "./identity"
import {
  GgufKey,
  GgufMetadata,
  normalizeParameterCount,
  projectGgufMetadata,
} from "./gguf-metadata"
import { makeGgufReader, type GgufReaderApi } from "./gguf-reader"
import type {
  InspectedModelArtifact,
  ModelFileFormat,
  ModelFileMetadata,
  ModelFileRole,
  SourceFileEntry,
  SourceFileSet,
} from "./types"
import { ModelFileWarning, ModelFormatError } from "./types"
import type { SchemaIssue } from "../schema-issues"

type GgufPartTopology = Data.TaggedEnum<{
  Unsplit: Record<never, never>
  Split: {
    readonly index: number
    readonly count: number
  }
}>

const GgufPartTopology = Data.taggedEnum<GgufPartTopology>()

interface ParsedGgufPart {
  readonly entry: SourceFileEntry
  readonly metadata: ModelFileMetadata
  readonly topology: GgufPartTopology
  readonly generalType: Option.Option<string>
}

type SplitGgufPart = ParsedGgufPart & {
  readonly topology: Extract<GgufPartTopology, { readonly _tag: "Split" }>
}

const formatId = ModelFileFormatId.make("gguf")
const formatError = (
  file: SourceFileKey,
  operation: ModelFormatError["operation"],
  reason: ModelFormatError["reason"],
  diagnostic: Option.Option<string> = Option.none(),
  issues: readonly SchemaIssue[] = [],
) => new ModelFormatError({ format: formatId, file, operation, reason, diagnostic, issues })

const recognizeGguf = (
  fs: FileSystem.FileSystem,
  entry: SourceFileEntry,
): Effect.Effect<boolean, ModelFormatError> => Effect.gen(function* () {
  const bytes = yield* Effect.scoped(
    fs.open(entry.path).pipe(Effect.flatMap((file) => file.readAlloc(8))),
  ).pipe(
    Effect.mapError(() => formatError(entry.key, "recognize", "unreadable")),
  )

  if (Option.isNone(bytes) || bytes.value.byteLength < 8) return false

  const header = bytes.value
  const magic = new TextDecoder().decode(header.subarray(0, 4))
  if (magic !== "GGUF") return false

  const version = new DataView(
    header.buffer,
    header.byteOffset + 4,
    4,
  ).getUint32(0, true)

  return version === 2 || version === 3
})

const readTopology = (
  metadata: GgufMetadata,
): GgufPartTopology => {
  const split = Option.zipWith(
    metadata.finiteNumber(GgufKey.SplitIndex),
    metadata.finiteNumber(GgufKey.SplitCount),
    (index, count) => [index, count] as const,
  )

  return Option.match(split, {
    onNone: () => GgufPartTopology.Unsplit(),
    onSome: ([index, count]) => GgufPartTopology.Split({ index, count }),
  })
}

const parsePart = (
  reader: GgufReaderApi,
  entry: SourceFileEntry,
): Effect.Effect<ParsedGgufPart, ModelFormatError> => reader.read(entry).pipe(
  Effect.map((document) => {
    const metadata = new GgufMetadata(document.typedMetadata)
    const projected = projectGgufMetadata(
      metadata,
      normalizeParameterCount(Option.fromNullable(document.parameterCount)),
    )

    return {
      entry,
      metadata: projected,
      topology: readTopology(metadata),
      generalType: metadata.string(GgufKey.GeneralType),
    }
  }),
  Effect.mapError((error) => error._tag === "GgufForeignLibraryError"
    ? formatError(entry.key, "decode", "foreign-library-failure", Option.some(error.diagnostic))
    : formatError(entry.key, "decode", "invalid-metadata", Option.none(), error.issues)),
)

const inferredRole = (
  part: ParsedGgufPart,
  primary: ParsedGgufPart,
  projectors: ReadonlySet<SourceFileKey>,
): ModelFileRole => Option.getOrElse(part.entry.declaredRole, () => {
  if (projectors.has(part.entry.key)) return "projector"
  return part === primary ? "primary" : "shard"
})

const duplicateWarnings = (
  parts: readonly ParsedGgufPart[],
): readonly ModelFileWarning[] => {
  const seen = new Set<SourceFileKey>()
  const duplicates = new Set<SourceFileKey>()

  for (const part of parts) {
    if (seen.has(part.entry.key)) duplicates.add(part.entry.key)
    else seen.add(part.entry.key)
  }

  return [...duplicates].map((key) => ModelFileWarning.DuplicatePart({ key }))
}

const relationshipWarnings = (
  parts: readonly ParsedGgufPart[],
  projectors: ReadonlySet<SourceFileKey>,
): readonly ModelFileWarning[] => parts.flatMap((part) => pipe(
  part.entry.declaredRole,
  Option.filter((role) => projectors.has(part.entry.key) && role !== "projector"),
  Option.match({
    onNone: () => [],
    onSome: (declaredRole) => [ModelFileWarning.RelationshipConflict({
      key: part.entry.key,
      declaredRole,
    })],
  }),
))

const assembleArtifact = (
  first: ParsedGgufPart,
  parts: readonly ParsedGgufPart[],
  artifactKey: Option.Option<ModelArtifactKey>,
  projectors: ReadonlySet<SourceFileKey> = new Set(),
): InspectedModelArtifact => {
  const primary = pipe(
    parts.find((part) => Option.contains(part.entry.declaredRole, "primary")),
    Option.fromNullable,
    Option.getOrElse(() => first),
  )

  return {
    key: Option.getOrElse(
      artifactKey,
      () => ModelArtifactKey.make(primary.entry.key),
    ),
    displayName: Option.getOrElse(
      primary.metadata.name,
      () => primary.entry.relativePath,
    ),
    parts: parts.map((part) => ({
      entry: part.entry,
      role: inferredRole(part, primary, projectors),
    })),
    metadata: primary.metadata,
    warnings: [
      ...duplicateWarnings(parts),
      ...relationshipWarnings(parts, projectors),
    ],
  }
}

const isSplitPart = (part: ParsedGgufPart): part is SplitGgufPart =>
  part.topology._tag === "Split"

const isStandaloneModel = (part: ParsedGgufPart): boolean =>
  part.topology._tag === "Unsplit" && !Option.contains(part.generalType, "mmproj")

const incompleteSplitWarning = (
  count: number,
  actual: number,
): ModelFileWarning => ModelFileWarning.IncompleteSplit({
  expectedParts: count,
  observedParts: actual,
})

const assembleSplitGroups = (
  parts: readonly SplitGgufPart[],
  artifactKey: Option.Option<ModelArtifactKey>,
): readonly InspectedModelArtifact[] => {
  const ungrouped = (): readonly InspectedModelArtifact[] => parts.map((part) => ({
    ...assembleArtifact(part, [part], Option.none()),
    warnings: [incompleteSplitWarning(part.topology.count, 1)],
  }))
  if (Option.isNone(artifactKey)) return ungrouped()
  const first = Option.fromNullable(parts[0])
  if (Option.isNone(first)) return []
  const count = first.value.topology.count
  const sameCount = parts.every((part) => part.topology.count === count)
  const indices = new Set(parts.map((part) => part.topology.index))
  const complete = Number.isSafeInteger(count)
    && count > 0
    && sameCount
    && parts.length === count
    && indices.size === count
    && [...indices].every((index) => Number.isSafeInteger(index) && index >= 0 && index < count)
  if (!complete) return ungrouped()
  const ordered = [...parts].sort((left, right) => left.topology.index - right.topology.index)
  return [assembleArtifact(first.value, ordered, artifactKey)]
}

const inspectSet = (
  reader: GgufReaderApi,
  recognize: (entry: SourceFileEntry) => Effect.Effect<boolean, ModelFormatError>,
  set: SourceFileSet,
): Effect.Effect<readonly InspectedModelArtifact[], ModelFormatError> => Effect.gen(function* () {
  const candidates = yield* Effect.forEach(
    set.entries,
    (entry) => recognize(entry).pipe(
      Effect.flatMap((recognized) => recognized
        ? parsePart(reader, entry).pipe(Effect.map(Option.some))
        : Effect.succeed(Option.none())),
    ),
    { concurrency: 4 },
  )

  const recognized = candidates.flatMap(Option.toArray)
  const declared = recognized.filter((part) => pipe(
    part.entry.declaredRole,
    Option.exists((role) => role === "primary" || role === "shard"),
  ))
  const firstDeclared = Option.fromNullable(declared[0])

  if (Option.isSome(firstDeclared)) {
    const declaredKeys = new Set(declared.map((part) => part.entry.key))
    const projectors = new Set(set.relationships.flatMap((relationship) =>
      declaredKeys.has(relationship.to) ? [relationship.from] : []))
    const related = recognized.filter((part) => projectors.has(part.entry.key))
    return [assembleArtifact(
      firstDeclared.value,
      [...declared, ...related],
      set.artifactKey,
      projectors,
    )]
  }

  return [
    ...recognized
      .filter(isStandaloneModel)
      .map((part) => assembleArtifact(part, [part], Option.none())),
    ...assembleSplitGroups(recognized.filter(isSplitPart), set.artifactKey),
  ]
})

const makeGgufFormatWithReader = (
  reader: GgufReaderApi,
): Effect.Effect<ModelFileFormat, never, FileSystem.FileSystem> => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const recognize = (entry: SourceFileEntry) => recognizeGguf(fs, entry)

  return {
    id: formatId,
    recognize,
    inspect: (set) => inspectSet(reader, recognize, set),
  }
})

export const makeGgufFormat = (): Effect.Effect<ModelFileFormat, never, FileSystem.FileSystem> => makeGgufFormatWithReader(makeGgufReader())
