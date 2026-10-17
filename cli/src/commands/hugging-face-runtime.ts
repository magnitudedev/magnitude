import {
  candidateDownloadBytes,
  catalogVariantOfName,
  formatStorageSize,
  modelVersionNotRunnableMessage,
  noRunnableVersionReason,
  preselectedDrafterCandidate,
  unsupportedReasonGroupMessage,
  unsupportedReasonGroups,
  type UnsupportedReasonGroup,
  unsupportedReasonMessages,
} from "@magnitudedev/client-common"
import {
  CatalogFormModelIdSchema,
  CustomFormModelIdSchema,
  DrafterCandidateSchema,
  HuggingFaceRepositoryIdSchema,
  HuggingFaceSearchResponseSchema,
  LocalModelSyncOutcomeSchema,
  Models,
  ModelVersionNotRunnable,
  type CandidateInspection,
  type CandidateVersion,
  type CatalogFormModelId,
  type DrafterCandidate,
  type DrafterChoice,
  type DrafterTarget,
  type HuggingFaceSearchResponse,
  type LocalModel,
  type LocalModelSyncOutcome,
  type UnsupportedReason,
} from "@magnitudedev/sdk"
import { Effect, Option, Schema } from "effect"
import { join } from "node:path"
import { renderDrafterChoices } from "./drafter-output"
import { catalogProducts, readCatalog, withClient, type CliModelsClient } from "./inference-client"
import { CandidateInspectionJsonSchema } from "./json-schemas"
import {
  absolutePath,
  isLocalPathReference,
  ModelCommandError,
  parseDrafterArgument,
  parsePullReference,
  type DrafterArgument,
  type HuggingFaceReference,
  type PullReference,
} from "./model-references"
import { ensureTrailingNewline, jsonDocument, renderTable, runCommand, asJson, type JsonOption } from "./output"

const plural = (count: number, singular: string, pluralForm = `${singular}s`) =>
  `${count} ${count === 1 ? singular : pluralForm}`

const quoted = (value: string) => JSON.stringify(value)

export const formatDownloads = (count: number): string => {
  const compact = (value: number, suffix: string) => `${Number(value.toFixed(value < 10 ? 1 : 0))}${suffix}`
  if (count < 1_000) return String(count)
  if (count < 1_000_000) return compact(count / 1_000, "K")
  return compact(count / 1_000_000, "M")
}

const variantOfLabel = (models: readonly LocalModel[], variantOf: Option.Option<CatalogFormModelId>): string =>
  Option.getOrElse(catalogVariantOfName(models, variantOf), () => Option.getOrElse(variantOf, () => "—"))

export interface SearchView {
  readonly query: string
  readonly all: boolean
  readonly response: HuggingFaceSearchResponse
  readonly models: readonly LocalModel[]
}

const reasonsLine = (reasons: readonly UnsupportedReason[]) => unsupportedReasonMessages(reasons).join(". ")

export const renderSearch = ({ query, all, response, models }: SearchView): string => {
  const supported = response.results.filter(({ unsupported }) => Option.isNone(unsupported))
  const unsupported = response.results.flatMap((result) => Option.match(result.unsupported, {
    onNone: () => [],
    onSome: (reasons) => [{ result, reasons }],
  }))
  const footer = response.hiddenCount > 0 && !all ? [
    `${plural(response.hiddenCount, "more result uses", "more results use")} architectures Magnitude doesn't run.`,
    `List them: magnitude catalog search ${quoted(query)} --all`,
  ] : []
  const supportedSection = supported.length === 0
    ? [`No Hugging Face models that run in Magnitude match ${quoted(query)}.`]
    : [
        `Hugging Face models matching ${quoted(query)}`,
        "",
        renderTable(supported, [
          { heading: "MODEL", value: ({ displayName }) => displayName },
          { heading: "UPLOADER", value: ({ uploader }) => uploader },
          { heading: "VARIANT OF", value: ({ variantOf }) => variantOfLabel(models, variantOf) },
          { heading: "DOWNLOADS", value: ({ downloads }) => formatDownloads(downloads) },
          { heading: "REPOSITORY", value: ({ repository }) => repository },
        ]).trimEnd(),
      ]
  const unsupportedSection = unsupported.length === 0 ? [] : [
    "",
    "Can't run in Magnitude",
    "",
    renderTable(unsupported, [
      { heading: "MODEL", value: ({ result }) => result.displayName },
      { heading: "REASON", value: ({ reasons }) => reasonsLine(reasons) },
      { heading: "REPOSITORY", value: ({ result }) => result.repository },
    ]).trimEnd(),
  ]
  return ensureTrailingNewline([
    ...supportedSection,
    ...unsupportedSection,
    ...(footer.length > 0 ? ["", ...footer] : []),
    ...(supported.length > 0 ? ["", "Inspect one: magnitude catalog inspect <repository>"] : []),
  ].join("\n"))
}

export const searchJson = jsonDocument(HuggingFaceSearchResponseSchema, ({ response }: SearchView) => response)

export const SEARCH_LIMIT_MAXIMUM = 100

const parseSearchLimit = (input: string | undefined) => {
  if (input === undefined) return Effect.succeed(Option.none<number>())
  const limit = Number(input)
  return Number.isSafeInteger(limit) && limit >= 1 && limit <= SEARCH_LIMIT_MAXIMUM
    ? Effect.succeed(Option.some(limit))
    : Effect.fail(new ModelCommandError({ message: `Limit must be an integer from 1 to ${SEARCH_LIMIT_MAXIMUM}` }))
}

export const readSearch = (
  client: CliModelsClient,
  query: string,
  limit: Option.Option<number>,
  all: boolean,
) => Effect.all({
  response: client.models.searchHuggingFaceModels({ query, limit, all }),
  catalog: readCatalog(client),
}).pipe(Effect.map(({ response, catalog }): SearchView => ({ query, all, response, models: catalogProducts(catalog) })))

export const searchHuggingFace = (
  query: string,
  options: JsonOption & { readonly all?: boolean; readonly limit?: string } = {},
) => runCommand({
  effect: Effect.gen(function* () {
    const trimmed = query.trim()
    if (trimmed.length === 0) return yield* new ModelCommandError({ message: "Search query must not be empty" })
    const limit = yield* parseSearchLimit(options.limit)
    const all = options.all === true
    return yield* withClient((client) => readSearch(client, trimmed, limit, all))
  }),
  render: renderSearch,
  json: asJson(options, searchJson),
})

/** The version a reference selected, or else the recommended one. */
export const chosenVersion = (inspection: CandidateInspection): Option.Option<CandidateVersion> =>
  Option.match(inspection.selectedVersion, {
    onSome: (label) => Option.fromNullable(inspection.versions.find((version) => version.label === label)),
    onNone: () => Option.fromNullable(inspection.versions.find(({ recommended }) => recommended)),
  })

const shortUnsupportedReason = (group: UnsupportedReasonGroup): string => {
  const { reason } = group
  switch (reason._tag) {
    case "WeightTypes": return `${reason.types.map(({ ggmlType }) => ggmlType).join(", ")} weights`
    case "Architecture": return `${reason.architecture} architecture`
    default: return unsupportedReasonGroupMessage(group)
  }
}

export const versionVerdict = (version: CandidateVersion, selected: boolean): string => {
  const verdict = Option.match(version.assessment, {
    onNone: () => "Not assessed",
    onSome: (assessment) => {
      switch (assessment._tag) {
        case "Fits": return "Fits"
        case "DoesNotFit": return "Too large"
        case "Unsupported": return `Unsupported: ${[...new Set(unsupportedReasonGroups(assessment.reasons).map(shortUnsupportedReason))].join(", ")}`
      }
    },
  })
  const marks = [
    ...(version.recommended ? ["recommended"] : []),
    ...(selected && !version.recommended ? ["selected"] : []),
  ]
  return [
    marks.length > 0 ? `${verdict} (${marks.join(", ")})` : verdict,
    ...(Option.isSome(version.onThisComputer) ? ["already on this computer"] : []),
  ].join(" · ")
}

export interface InspectionView {
  readonly inspection: CandidateInspection
  readonly drafters: readonly DrafterCandidate[]
  readonly models: readonly LocalModel[]
}

const inspectionHeader = (inspection: CandidateInspection, models: readonly LocalModel[]): string[] => {
  const details = [
    ...Option.match(inspection.variantOf, {
      onNone: () => [],
      onSome: () => [`Variant of ${variantOfLabel(models, inspection.variantOf)}`],
    }),
    ...Option.toArray(inspection.license),
    ...(inspection.source._tag === "HuggingFace" ? [`pinned to ${inspection.source.commit.slice(0, 7)}`] : []),
  ]
  return [inspection.displayName, ...(details.length > 0 ? [details.join(" · ")] : [])]
}

const isRunnable = (version: CandidateVersion) =>
  Option.exists(version.assessment, (assessment) => assessment._tag === "Fits")

export const acquisitionHint = (inspection: CandidateInspection): string => Option.match(
  Option.filter(chosenVersion(inspection), isRunnable), {
    onNone: () => ["No version of this model runs on this computer.", ...Option.toArray(noRunnableVersionReason(inspection.versions))].join(" "),
    onSome: (version) => inspection.source._tag === "HuggingFace"
      ? `Download: magnitude catalog pull ${inspection.source.repository}:${version.label}`
      : `Add: magnitude models import ${join(inspection.source.path, version.files[0]!)}`,
  })

/** "Download size: 3.7 GB (model 3.01 GB + projector 670 MB + drafter 685 MB)", omitting parts already here. */
export const downloadSizeLine = (
  inspection: CandidateInspection,
  version: CandidateVersion,
  drafters: readonly DrafterCandidate[],
): string => {
  const download = candidateDownloadBytes(version, inspection.projector, preselectedDrafterCandidate(drafters))
  if (download.totalBytes === 0) return "Download size: nothing to download, it's all on this computer."
  const parts = [
    ["model", download.modelBytes],
    ["projector", download.projectorBytes],
    ["drafter", download.drafterBytes],
  ] as const
  const present = parts.filter(([, bytes]) => bytes > 0)
  const total = `Download size: ${formatStorageSize(download.totalBytes)}`
  return present.length < 2
    ? `${total} (${present[0]![0]})`
    : `${total} (${present.map(([part, bytes]) => `${part} ${formatStorageSize(bytes)}`).join(" + ")})`
}

export const renderInspection = ({ inspection, drafters, models }: InspectionView): string => {
  const header = inspectionHeader(inspection, models)
  if (Option.isSome(inspection.unsupportedArchitecture)) return ensureTrailingNewline([
    ...header,
    "",
    `Magnitude can't run this model: ${reasonsLine(inspection.unsupportedArchitecture.value)}.`,
  ].join("\n"))
  if (inspection.versions.length === 0) return ensureTrailingNewline([
    ...header,
    "",
    "This repository has no GGUF files Magnitude can use.",
  ].join("\n"))
  const chosen = Option.getOrUndefined(chosenVersion(inspection))
  const selectedLabel = Option.getOrUndefined(inspection.selectedVersion)
  const versions = renderTable(inspection.versions, [
    { heading: "VERSION", value: ({ label }) => label },
    { heading: "SIZE", value: ({ sizeBytes }) => formatStorageSize(sizeBytes) },
    { heading: "ON THIS COMPUTER", value: (version) => versionVerdict(version, version.label === selectedLabel) },
  ]).trimEnd()
  const runnable = chosen !== undefined && isRunnable(chosen)
  const drafterSection = !runnable ? [] : [
    "",
    renderDrafterChoices(drafters, Option.map(preselectedDrafterCandidate(drafters), ({ reference }) => reference)),
  ]
  return ensureTrailingNewline([
    ...header,
    "",
    versions,
    ...drafterSection,
    "",
    ...(runnable ? [downloadSizeLine(inspection, chosen, drafters)] : []),
    acquisitionHint(inspection),
  ].join("\n"))
}

export const InspectionJsonSchema = Schema.Struct({
  inspection: CandidateInspectionJsonSchema,
  drafters: Schema.Array(DrafterCandidateSchema),
})

export const inspectionJson = jsonDocument(InspectionJsonSchema, ({ inspection, drafters }: InspectionView) => ({
  inspection,
  drafters,
}))

/** What the service is asked to inspect: Hugging Face forms as typed, local paths made absolute. */
export const candidateReference = (input: string): string =>
  isLocalPathReference(input) ? absolutePath(input) : input

const candidateDrafterTarget = (
  inspection: CandidateInspection,
  version: CandidateVersion,
): DrafterTarget => ({ _tag: "Candidate", source: inspection.source, files: version.files })

const listCandidateDrafters = (client: CliModelsClient, inspection: CandidateInspection) =>
  Option.match(Option.filter(chosenVersion(inspection), isRunnable), {
    onNone: () => Effect.succeed<readonly DrafterCandidate[]>([]),
    onSome: (version) => client.models.listDrafterCandidates({
      target: candidateDrafterTarget(inspection, version),
      other: Option.none(),
    }).pipe(Effect.map(({ candidates }) => candidates)),
  })

export const readInspection = (client: CliModelsClient, reference: string) => Effect.gen(function* () {
  const [inspection, catalog] = yield* Effect.all([
    client.models.inspectModelCandidate({ reference }),
    readCatalog(client),
  ])
  const drafters = yield* listCandidateDrafters(client, inspection)
  return { inspection, drafters, models: catalogProducts(catalog) } satisfies InspectionView
})

export const inspectCandidate = (input: string, options: JsonOption = {}) => runCommand({
  effect: Effect.gen(function* () {
    const reference = input.trim()
    if (reference.length === 0) {
      return yield* new ModelCommandError({ message: "Expected a Hugging Face link, owner/repo[:QUANT], or a GGUF path" })
    }
    return yield* withClient((client) => readInspection(client, candidateReference(reference)))
  }),
  render: renderInspection,
  json: asJson(options, inspectionJson),
})

/** The drafter choice an argument names without asking the service. */
export const drafterChoice = (argument: Exclude<DrafterArgument, { readonly _tag: "Lookup" }>): DrafterChoice => {
  switch (argument._tag) {
    case "Auto": return { _tag: "Auto" }
    case "None": return { _tag: "None" }
    case "Reference": return { _tag: "Reference", reference: argument.reference }
  }
}

const HUB_URL = /^https?:\/\/(?:www\.)?huggingface\.co\/([^/]+\/[^/]+)\/(?:blob|resolve)\/[^/]+\/(.+)$/
const HUB_REFERENCE = /^([^/@:\s]+\/[^/@:\s]+)(?:@[^:]+)?:(.+)$/
const HUB_FILE = /^(?:hf:)?([^/@:\s]+\/[^/@:\s]+)\/([^:]+)$/i

/** Whether a typed drafter reference names an already listed candidate: the same Hub file or the same local path. */
export const drafterReferenceMatches = (input: string, reference: DrafterCandidate["reference"]): boolean => {
  const typed = input.trim()
  switch (reference._tag) {
    case "HuggingFace": {
      const hub = HUB_URL.exec(typed) ?? HUB_REFERENCE.exec(typed) ?? HUB_FILE.exec(typed)
      return hub !== null
        && hub[1]!.toLowerCase() === reference.repository.toLowerCase()
        && decodeURIComponent(hub[2]!) === reference.path
    }
    case "Local":
      return absolutePath(typed) === reference.path
    case "Catalog":
    case "Embedded":
      return false
  }
}

const lookUpDrafter = (
  client: CliModelsClient,
  target: DrafterTarget,
  reference: string,
): Effect.Effect<DrafterChoice, unknown> => client.models.listDrafterCandidates({
  target,
  other: Option.some(reference),
}).pipe(Effect.flatMap(({ candidates }) => {
  const found = candidates.find(({ category }) => category === "Other")
    ?? candidates.find((candidate) => drafterReferenceMatches(reference, candidate.reference))
  if (found === undefined) {
    return Effect.fail(new ModelCommandError({ message: `Magnitude couldn't find a drafter at ${reference}.` }))
  }
  if (found.verdict._tag === "Rejected") {
    return Effect.fail(new ModelCommandError({
      message: `${reference} can't be used as a drafter for this model: ${reasonsLine(found.verdict.reasons)}.`,
    }))
  }
  return Effect.succeed<DrafterChoice>({ _tag: "Reference", reference: found.reference })
}))

/** Resolves a drafter argument for a target, refusing a looked-up drafter the engine rejects. */
export const resolveDrafterChoice = (
  client: CliModelsClient,
  target: DrafterTarget,
  argument: DrafterArgument,
): Effect.Effect<DrafterChoice, unknown> => argument._tag === "Lookup"
  ? lookUpDrafter(client, target, argument.reference)
  : Effect.succeed(drafterChoice(argument))

const AddedModelSchema = Models.addHuggingFaceModel.successSchema

export const PullResultSchema = Schema.Union(
  Schema.TaggedStruct("Catalog", {
    modelId: CatalogFormModelIdSchema,
    outcome: LocalModelSyncOutcomeSchema,
  }),
  Schema.TaggedStruct("Custom", {
    modelId: CustomFormModelIdSchema,
    outcome: LocalModelSyncOutcomeSchema,
  }),
  Schema.TaggedStruct("HuggingFace", {
    repository: HuggingFaceRepositoryIdSchema,
    versionRequested: Schema.Boolean,
    added: AddedModelSchema,
  }),
)
export type PullResult = typeof PullResultSchema.Type

const syncStarted = (modelId: string, outcome: Exclude<LocalModelSyncOutcome, "AlreadyCurrent">): string => {
  const started = {
    Installing: `Downloading ${modelId}.`,
    Updating: `Updating ${modelId}.`,
    DownloadingDrafter: `Downloading the drafter for ${modelId}.`,
  }[outcome]
  return `${started}\nCheck progress: magnitude models status ${modelId}\n`
}

export const renderPull = (result: PullResult): string => {
  if (result._tag === "Catalog") return result.outcome === "AlreadyCurrent"
    ? `${result.modelId} is already installed and up to date.\n`
    : syncStarted(result.modelId, result.outcome)
  if (result._tag === "Custom") return result.outcome === "AlreadyCurrent"
    ? `${result.modelId} is up to date.\n`
    : syncStarted(result.modelId, result.outcome)
  const { added } = result
  const name = `${result.repository}:${added.version}`
  const choice = result.versionRequested ? "" : " (the recommended version)"
  const drafter = `Drafter: ${Option.match(added.drafter, { onNone: () => "none", onSome: ({ name }) => name })}`
  const progress = `Check progress: magnitude models status ${added.modelId}`
  if (added.outcome === "Started") return ensureTrailingNewline([
    `Downloading ${name}${choice}.`,
    drafter,
    `Model ID: ${added.modelId}`,
    progress,
  ].join("\n"))
  return ensureTrailingNewline([
    `${name}${choice} is already on this computer. ${added.drafterDownloadStarted ? "Downloading only the drafter." : "Nothing to download."}`,
    drafter,
    `Model ID: ${added.modelId}`,
    added.drafterDownloadStarted ? progress : `Load it: magnitude models load ${added.modelId}`,
  ].join("\n"))
}

export const pullJson = jsonDocument(PullResultSchema, (result: PullResult) => result)

/** A refused version in the CLI's words; one that only lacks memory can be forced. */
const versionRefusal = (error: ModelVersionNotRunnable): ModelCommandError => new ModelCommandError({
  message: error.verdict._tag === "DoesNotFit"
    ? `${modelVersionNotRunnableMessage(error)} Download it anyway with: magnitude catalog pull ${error.reference} --force`
    : modelVersionNotRunnableMessage(error),
})

const addHuggingFaceModel = (
  client: CliModelsClient,
  input: string,
  reference: HuggingFaceReference,
  drafter: DrafterArgument,
  force: boolean,
): Effect.Effect<PullResult, unknown> => Effect.gen(function* () {
  const versionRequested = Option.isSome(reference.version) || Option.isSome(reference.file)
  if (Option.isNone(reference.file) && drafter._tag !== "Lookup") {
    const added = yield* client.models.addHuggingFaceModel({
      repository: reference.repository,
      revision: reference.revision,
      version: reference.version,
      drafter: drafterChoice(drafter),
      force,
    })
    return { _tag: "HuggingFace" as const, repository: reference.repository, versionRequested, added }
  }
  const inspection = yield* client.models.inspectModelCandidate({ reference: input })
  if (inspection.source._tag !== "HuggingFace") {
    return yield* new ModelCommandError({ message: `${input} is not a Hugging Face repository.` })
  }
  const version = yield* Option.match(chosenVersion(inspection), {
    onNone: () => Effect.fail(new ModelCommandError({
      message: `No version of ${reference.repository} runs on this computer. See: magnitude catalog inspect ${reference.repository}`,
    })),
    onSome: Effect.succeed,
  })
  const added = yield* client.models.addHuggingFaceModel({
    repository: inspection.source.repository,
    revision: Option.some(inspection.source.commit),
    version: Option.some(version.label),
    drafter: yield* resolveDrafterChoice(client, candidateDrafterTarget(inspection, version), drafter),
    force,
  })
  return { _tag: "HuggingFace" as const, repository: inspection.source.repository, versionRequested, added }
}).pipe(Effect.catchIf(
  (error): error is ModelVersionNotRunnable => error instanceof ModelVersionNotRunnable,
  (error) => Effect.fail(versionRefusal(error)),
))

export const pull = (
  client: CliModelsClient,
  input: string,
  reference: PullReference,
  drafter: DrafterArgument,
  force = false,
): Effect.Effect<PullResult, unknown> => {
  switch (reference._tag) {
    case "Catalog": return client.models.syncLocalModel({ modelId: reference.modelId }).pipe(
      Effect.map(({ outcome }) => ({ _tag: "Catalog" as const, modelId: reference.modelId, outcome })))
    case "Custom": return client.models.syncLocalModel({ modelId: reference.modelId }).pipe(
      Effect.map(({ outcome }) => ({ _tag: "Custom" as const, modelId: reference.modelId, outcome })))
    case "HuggingFace": return addHuggingFaceModel(client, input, reference.reference, drafter, force)
  }
}

export const pullModel = (
  modelInput: string,
  options: JsonOption & { readonly drafter?: string; readonly force?: boolean } = {},
) => runCommand({
  effect: Effect.gen(function* () {
    const reference = yield* parsePullReference(modelInput.trim())
    if (reference._tag === "Catalog" && options.drafter !== undefined) {
      return yield* new ModelCommandError({
        message: "Catalog models come with their drafter; --drafter applies only to models from Hugging Face.",
      })
    }
    if (reference._tag === "Custom" && options.drafter !== undefined) {
      return yield* new ModelCommandError({
        message: `Change this model's drafter with: magnitude models drafter ${reference.modelId} <auto | none | drafter>`,
      })
    }
    const drafter = yield* parseDrafterArgument(options.drafter ?? "auto")
    if (reference._tag !== "HuggingFace" && options.force === true) {
      return yield* new ModelCommandError({ message: "--force applies only to models from Hugging Face." })
    }
    return yield* withClient((client) => pull(client, modelInput.trim(), reference, drafter, options.force === true))
  }),
  render: renderPull,
  json: asJson(options, pullJson),
})
