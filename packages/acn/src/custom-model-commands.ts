import * as FileSystem from "@effect/platform/FileSystem"
import { Array as Arr, Context, Effect, Either, Layer, Option, Schedule, Schema, Stream } from "effect"
import { dirname, basename, isAbsolute, normalize, sep } from "node:path"
import {
  CandidateInspectionSchema,
  CatalogFormModelIdSchema,
  CustomFormModelIdSchema,
  DirectoryAccessDenied,
  DirectoryNotFound,
  DrafterNotUsable,
  HuggingFaceRepositoryIdSchema,
  HuggingFaceSearchResultSchema,
  InvalidDirectoryPath,
  InvalidModelReference,
  LocalModelAssessmentSchema,
  LocalModelMutationFailed,
  ModelRepositoryNotFound,
  ModelSourceUnavailable,
  ModelVersionNotRunnable,
  PathNotDirectory,
  parseModelId,
  preselectedDrafterCandidate,
  type AutomaticModelLocationKind,
  type CandidateInspection,
  type CatalogFormModelId,
  type CustomFormModelId,
  type CustomLocalModel,
  type AddedModel,
  type CustomModelMutationError,
  type DrafterCandidate,
  type DrafterChoice,
  type DrafterReference,
  type DrafterTarget,
  type HuggingFaceRepositoryId,
  type HuggingFaceSearchResponse,
  type LocalInferenceError,
  type LocalModelSyncOutcome,
  type ModelCandidateError,
  type ModelId,
  type ModelLocation,
  type ModelLocationError,
  type ModelLocationKind,
} from "@magnitudedev/acn-protocol"
import {
  bundlePackages,
  IcnCandidates,
  IcnDownloads,
  IcnLocalModels,
  IcnModelAssessments,
  packageHoldsFile,
} from "@magnitudedev/icn"
import type * as Native from "@magnitudedev/icn-protocol/schemas"
import type { GeneratedClientError } from "@magnitudedev/openapi-effect/client-runtime"
import type { CustomModelDeclaration, CustomModelsConfig, PersistedDrafterChoice } from "@magnitudedev/storage"
import { CatalogModelSync } from "./catalog-model-sync"
import { CustomModels } from "./custom-models"
import { CustomModelReconciler } from "./custom-model-reconciler"
import {
  currentDownload,
  customModelDownloads,
  downloadHasUnacknowledgedFailure,
  downloadIsActive,
  drafterPackage,
} from "./custom-model-downloads"
import {
  DrafterMethods,
  drafterReferenceKey,
  drafterReferenceNaming,
  nativeDrafterReference,
  presentField,
  productDrafterCandidate,
} from "./drafter-references"
import { LocalModelSources } from "./local-model-sources"
import { LocalModels, projectAssessment } from "./local-models"
import { ModelLocations } from "./model-locations"
import { drafterInput, parseModelReference, repositoryDisplayName } from "./model-references"
import { productUnsupportedReasons } from "./unsupported-reasons"


/** A drafter choice and whether choosing it started a drafter-only download. */
export interface ChosenDrafter {
  readonly drafter: Option.Option<DrafterCandidate>
  readonly drafterDownloadStarted: boolean
}

/** Where a catalog model found only outside Magnitude's store keeps its files. */
export interface ExternalModelFiles {
  readonly path: string
  readonly location: Option.Option<ModelLocationKind>
}

const locationPhrase = (files: ExternalModelFiles): string => Option.match(files.location, {
  onNone: () => files.path,
  onSome: (kind) => {
    switch (kind) {
      case "HuggingFaceCache": return `your Hugging Face cache (${files.path})`
      case "LmStudio": return `LM Studio's models folder (${files.path})`
      case "LlamaCpp": return `the llama.cpp cache (${files.path})`
      case "Ollama": return `Ollama's models folder (${files.path})`
      case "Folder": return files.path
    }
  },
})

const deleteElsewhere = (location: Option.Option<ModelLocationKind>): string => Option.match(location, {
  onNone: () => "or delete the files yourself",
  onSome: (kind) => {
    switch (kind) {
      case "HuggingFaceCache": return "or delete it with the Hugging Face CLI (`hf cache delete`)"
      case "LmStudio": return "or delete it in LM Studio"
      case "Ollama": return "or delete it with `ollama rm`"
      case "LlamaCpp":
      case "Folder": return "or delete the files yourself"
    }
  },
})

/** Why Remove leaves a catalog model found outside Magnitude's store, and what to do instead. */
export const externalCatalogRemovalRefusal = (modelId: CatalogFormModelId, files: ExternalModelFiles) =>
  new LocalModelMutationFailed({
    code: "model_removal_retained_external",
    message: `${modelId} uses files in ${locationPhrase(files)}, which Magnitude never deletes. `
      + `Hide it with \`magnitude models hide ${modelId}\`, ${deleteElsewhere(files.location)}.`,
    retryable: false,
  })

const catalogHideRefusal = (modelId: ModelId) => new LocalModelMutationFailed({
  code: "catalog_model_not_hideable",
  message: `${modelId} is a catalog model. Remove it with \`magnitude catalog remove ${modelId}\` instead.`,
  retryable: false,
})

const managedHideRefusal = (modelId: ModelId, origin: "Added" | "Imported") => new LocalModelMutationFailed({
  code: "managed_model_not_hideable",
  message: origin === "Added"
    ? `${modelId} was added to Magnitude, so it can't be hidden. Remove it with \`magnitude catalog remove ${modelId}\` instead.`
    : `${modelId} was imported into Magnitude, so it can't be hidden. Remove it with \`magnitude catalog remove ${modelId}\` instead.`,
  retryable: false,
})

/** The location holding a path: the one with the deepest scanned directory that contains it. */
const containingLocation = (path: string, locations: readonly ModelLocation[]): Option.Option<ModelLocationKind> =>
  Option.fromNullable(locations
    .flatMap((location) => location.paths.map((directory) => ({ kind: location.kind, directory: directory.path })))
    .filter(({ directory }) => path === directory || path.startsWith(`${directory.replace(/[\\/]+$/, "")}${sep}`))
    .sort((left, right) => right.directory.length - left.directory.length)[0]?.kind)

/** Custom-model commands: finding and adding models, drafters, hiding, and model locations. */
export interface CustomModelCommandsApi {
  readonly searchHuggingFace: (request: {
    readonly query: string
    readonly limit: Option.Option<number>
    readonly all: boolean
  }) => Effect.Effect<HuggingFaceSearchResponse, ModelCandidateError>
  readonly inspectCandidate: (reference: string) => Effect.Effect<CandidateInspection, ModelCandidateError>
  readonly listDrafterCandidates: (
    target: DrafterTarget,
    other: Option.Option<string>,
  ) => Effect.Effect<readonly DrafterCandidate[], ModelCandidateError>
  readonly addHuggingFaceModel: (request: {
    readonly repository: HuggingFaceRepositoryId
    readonly revision: Option.Option<string>
    readonly version: Option.Option<string>
    readonly drafter: DrafterChoice
    readonly force: boolean
  }) => Effect.Effect<AddedModel, CustomModelMutationError>
  readonly importModelFile: (path: string, drafter: DrafterChoice) => Effect.Effect<
    ChosenDrafter & { readonly modelId: CustomFormModelId; readonly outcome: "Started" | "AlreadyOnThisComputer" },
    CustomModelMutationError
  >
  readonly setModelDrafter: (
    modelId: CustomFormModelId,
    drafter: DrafterChoice,
  ) => Effect.Effect<ChosenDrafter, CustomModelMutationError>
  readonly hide: (modelId: ModelId) => Effect.Effect<void, LocalModelMutationFailed>
  readonly unhide: (modelId: ModelId) => Effect.Effect<void, LocalModelMutationFailed>
  readonly externalCatalogFiles: (modelId: CatalogFormModelId) => Effect.Effect<Option.Option<ExternalModelFiles>>
  readonly addFolder: (path: string) => Effect.Effect<void, ModelLocationError>
  readonly removeFolder: (path: string) => Effect.Effect<void, ModelLocationError>
  readonly setLocationEnabled: (
    kind: AutomaticModelLocationKind,
    enabled: boolean,
  ) => Effect.Effect<void, LocalModelMutationFailed>
  readonly listLocations: Effect.Effect<readonly ModelLocation[]>
  readonly remove: (modelId: CustomFormModelId) => Effect.Effect<{}, LocalInferenceError>
  readonly sync: (modelId: CustomFormModelId) => Effect.Effect<{ readonly outcome: LocalModelSyncOutcome }, LocalInferenceError>
  readonly cancelSync: (modelId: CustomFormModelId) => Effect.Effect<{}, LocalInferenceError>
  readonly acknowledgeSyncFailure: (modelId: CustomFormModelId) => Effect.Effect<{}, LocalInferenceError>
}
export class CustomModelCommands extends Context.Tag("CustomModelCommands")<
  CustomModelCommands,
  CustomModelCommandsApi
>() {}


const remoteMessage = (cause: GeneratedClientError<unknown>): string => {
  if (cause._tag === "GeneratedClientRemoteError") {
    const body = cause.body as { readonly error?: { readonly message?: unknown } } | undefined
    if (typeof body?.error?.message === "string" && body.error.message.length > 0) return body.error.message
  }
  return "The inference service could not read this model."
}

/** The product message for a repository whose terms must be accepted with an access token. */
export const accessRequiredMessage = "This repository requires accepting its terms on Hugging Face and an access token. "
  + "Set HF_TOKEN to a token from an account that has accepted them, then try again."

/** The product message for a repository Hugging Face won't show: missing, private or gated. */
export const repositoryNotFoundMessage = "Repository not found, or it's private or gated. "
  + "Check the name, or set HF_TOKEN if you have access."

/** The product message for a file whose contents Magnitude couldn't read. */
export const unreadableFileMessage = "Magnitude couldn't read this file. It may be incomplete, damaged, "
  + "or use a format Magnitude doesn't support yet."

const remoteCode = (cause: GeneratedClientError<unknown>): string | undefined => {
  if (cause._tag !== "GeneratedClientRemoteError") return undefined
  const code = (cause.body as { readonly error?: { readonly code?: unknown } } | undefined)?.error?.code
  return typeof code === "string" ? code : undefined
}

/**
 * A rejected reference is the caller's to fix. A repository that needs access and a file that
 * can't be read get product messages; other Hugging Face and file failures keep the inference
 * service's own message. Only a connection that never reached it is a service reachability
 * failure, and a response that doesn't match the contract is never presented as one.
 */
const candidateFailure = (reference: string) => (cause: GeneratedClientError<unknown>): ModelCandidateError => {
  switch (cause._tag) {
    case "GeneratedClientRemoteError": {
      if (cause.status === 400 || cause.status === 404) return new InvalidModelReference({ reference, message: remoteMessage(cause) })
      switch (remoteCode(cause)) {
        case "hugging_face_access_required":
          return new ModelSourceUnavailable({ reason: "AccessRequired", message: accessRequiredMessage, retryable: false })
        case "hugging_face_repository_not_found":
          return new ModelRepositoryNotFound({ reference, message: repositoryNotFoundMessage })
        case "integrity_failed":
          return new ModelSourceUnavailable({ reason: "Unreadable", message: unreadableFileMessage, retryable: false })
        default: {
          const retryable = cause.status === 409 || cause.status >= 500
          return new ModelSourceUnavailable({
            reason: retryable ? "SourceUnreachable" : "Unavailable",
            message: remoteMessage(cause),
            retryable,
          })
        }
      }
    }
    case "GeneratedClientTransportError":
    case "GeneratedClientIncompleteStreamError":
      return new ModelSourceUnavailable({
        reason: "ServiceUnreachable",
        message: "Magnitude couldn't reach its inference service. Try again in a moment.",
        retryable: true,
      })
    case "GeneratedClientInputError":
    case "GeneratedClientInvalidResponseError":
      return new ModelSourceUnavailable({
        reason: "Unreadable",
        message: "The inference service could not read this model.",
        retryable: false,
      })
  }
}

const notFound = (modelId: ModelId) => new LocalModelMutationFailed({
  code: "custom_model_not_found",
  message: `${modelId} is not a model on this computer.`,
  retryable: false,
})

const commandFailed = (operation: string) => (cause: GeneratedClientError<unknown> | ModelCandidateError) =>
  cause._tag === "InvalidModelReference" || cause._tag === "ModelRepositoryNotFound" || cause._tag === "ModelSourceUnavailable"
    ? new LocalModelMutationFailed({ code: `custom_model_${operation}_failed`, message: cause.message, retryable: true })
    : new LocalModelMutationFailed({
        code: `custom_model_${operation}_failed`,
        message: remoteMessage(cause),
        retryable: cause._tag !== "GeneratedClientRemoteError" || cause.status >= 500,
      })

const withoutKey = <K extends string, V>(record: Readonly<Record<K, V>>, key: K): Record<K, V> => {
  const next = { ...record }
  delete next[key]
  return next
}

/** Just after a restart no download exists, so a missing part of a model was interrupted by it. */
export const interruptedByRestart = (model: CustomLocalModel): boolean =>
  (model.origin._tag === "Added" && model.acquisitionState._tag === "NotInstalled")
  || Option.exists(model.drafter, ({ availability }) => availability._tag === "DownloadFailed")

/** A resolved candidate pinned with the method the engine binds it with; nothing binding keeps Auto. */
const pinnedChoice = (requested: DrafterChoice, chosen: Option.Option<DrafterCandidate>): PersistedDrafterChoice =>
  requested._tag === "None"
    ? { _tag: "None" }
    : Option.match(Option.filter(chosen, ({ verdict }) => verdict._tag === "Bound"), {
        onNone: (): PersistedDrafterChoice => ({ _tag: "Auto" }),
        onSome: ({ reference, verdict }): PersistedDrafterChoice => verdict._tag === "Bound"
          ? { _tag: "Reference", reference, method: verdict.method }
          : { _tag: "Auto" },
      })

/**
 * The drafter choice persisted for a model: Auto stays Auto, as does picking the recommended drafter, so a
 * later catalog drafter still applies; any other reference is pinned with its method.
 */
const persistedChoice = (
  requested: DrafterChoice,
  chosen: Option.Option<DrafterCandidate>,
  recommended: Option.Option<DrafterReference>,
): PersistedDrafterChoice => {
  const recommendedPicked = requested._tag === "Reference" && Option.exists(recommended,
    (reference) => drafterReferenceKey(reference) === drafterReferenceKey(requested.reference))
  return requested._tag === "Auto" || recommendedPicked ? { _tag: "Auto" } : pinnedChoice(requested, chosen)
}

export const CustomModelCommandsLive: Layer.Layer<
  CustomModelCommands,
  never,
  | IcnCandidates
  | IcnDownloads
  | IcnModelAssessments
  | IcnLocalModels
  | LocalModelSources
  | LocalModels
  | CustomModels
  | CustomModelReconciler
  | DrafterMethods
  | ModelLocations
  | CatalogModelSync
  | FileSystem.FileSystem
> = Layer.scoped(CustomModelCommands, Effect.gen(function* () {
  const icn = yield* IcnCandidates
  const downloads = yield* IcnDownloads
  const assessments = yield* IcnModelAssessments
  const sources = yield* LocalModelSources
  const localModels_ = yield* IcnLocalModels
  const localModels = yield* LocalModels
  const customModels = yield* CustomModels
  const reconciler = yield* CustomModelReconciler
  const methods = yield* DrafterMethods
  const locations = yield* ModelLocations
  const catalogSync = yield* CatalogModelSync
  const fs = yield* FileSystem.FileSystem

  const environmentId = assessments.get.pipe(Effect.map(({ state }) => state.environmentId))

  const parse = (reference: string) => Either.match(parseModelReference(reference, locations.environment), {
    onLeft: (message) => Effect.fail(new InvalidModelReference({ reference, message })),
    onRight: Effect.succeed,
  })

  const productInspection = (
    native: Native.CandidateInspection,
    selection: { readonly file: Option.Option<string>; readonly quant: Option.Option<string> },
    reference: string,
  ) => Effect.gen(function* () {
    const environment = yield* environmentId
    const source = native.source
    const versions = yield* Effect.forEach(native.versions, (version) => Effect.gen(function* () {
      const decoded = Schema.decodeUnknownOption(CustomFormModelIdSchema)(version.modelId)
      if (Option.isNone(decoded)) {
        return yield* new ModelSourceUnavailable({
          reason: "Unreadable",
          message: "The inference service could not identify this model.",
          retryable: false,
        })
      }
      return {
        label: version.label,
        modelId: decoded.value,
        files: version.files.map(({ path }) => path),
        sizeBytes: version.sizeBytes,
        downloadBytes: version.downloadBytes,
        assessment: Option.map(version.assessment, (assessment) => projectAssessment(environment, assessment)),
        recommended: version.recommended,
        catalogModel: Option.flatMap(version.catalogModel, Schema.decodeUnknownOption(CatalogFormModelIdSchema)),
        onThisComputer: version.onThisComputer,
      }
    }))
    const named = Option.orElse(
      Option.flatMap(selection.file, (file) => Option.fromNullable(versions.find(({ files }) => files.includes(file)))),
      () => Option.flatMap(selection.quant, (quant) => Option.fromNullable(
        versions.find(({ label }) => label.toLowerCase() === quant.toLowerCase())
          ?? versions.find(({ files }) => files[0]!.toLowerCase().includes(quant.toLowerCase())),
      )),
    )
    if ((Option.isSome(selection.file) || Option.isSome(selection.quant)) && Option.isNone(named)) {
      return yield* new InvalidModelReference({ reference, message: "This repository has no such version." })
    }
    const selected = Option.orElse(named, () => Option.fromNullable(versions.find(({ recommended }) => recommended)))
    return yield* Schema.decodeUnknown(CandidateInspectionSchema)({
      source,
      displayName: native.displayName.trim().length > 0
        ? native.displayName
        : source._tag === "HuggingFace" ? repositoryDisplayName(source.repository) : basename(source.path),
      ...presentField("license", native.license),
      ...presentField("variantOf", native.variantOf),
      versions: versions.map(({ assessment, catalogModel, onThisComputer, ...version }) => ({
        ...version,
        ...presentField("assessment", Option.map(assessment, (value) => Schema.encodeSync(LocalModelAssessmentSchema)(value))),
        ...presentField("catalogModel", catalogModel),
        ...presentField("onThisComputer", onThisComputer),
      })),
      ...presentField("selectedVersion", Option.map(selected, ({ label }) => label)),
      ...presentField("projector", native.projector),
      ...presentField("unsupportedArchitecture", Option.filter(native.unsupportedArchitecture, (reasons) => reasons.length > 0)
        .pipe(Option.map((reasons) => productUnsupportedReasons(reasons, "")))),
    }).pipe(Effect.mapError(() => new ModelSourceUnavailable({
      reason: "Unreadable",
      message: "The inference service described this model in a form Magnitude doesn't understand.",
      retryable: false,
    })))
  })

  const inspect = (reference: string) => Effect.gen(function* () {
    const parsed = yield* parse(reference)
    const native = yield* icn.inspect(parsed._tag === "HuggingFace"
      ? { _tag: "HuggingFace", repository: parsed.repository, revision: parsed.revision }
      : { _tag: "Local", path: parsed.path },
    ).pipe(Effect.mapError(candidateFailure(reference)))
    return yield* productInspection(
      native,
      parsed._tag === "HuggingFace" ? parsed : { file: Option.none(), quant: Option.none() },
      reference,
    )
  })

  /** The ICN drafter target and, when its files are readable, the same target as a candidate for bundles. */
  interface NativeTarget {
    readonly request: (other: Option.Option<string>) => Native.DrafterCandidatesRequest
    readonly candidate: Option.Option<Native.ModelCandidate>
  }

  const nativeTarget = (target: DrafterTarget): Effect.Effect<NativeTarget> => Effect.gen(function* () {
    if (target._tag === "Candidate") {
      const candidate: Native.ModelCandidate = {
        source: target.source as Native.CandidateSource,
        target: [...target.files],
        projector: Option.none(),
        drafter: Option.none(),
      }
      return {
        request: (other: Option.Option<string>) =>
          ({ _tag: "Candidate", candidate, other }) as Native.DrafterCandidatesRequest,
        candidate: Option.some(candidate),
      }
    }
    const custom = (yield* sources.state).customModels.find(({ id }) => id === target.modelId)
    const declaration = (yield* customModels.get).declarations[target.modelId as CustomFormModelId]
    // An added model still downloading has no files yet; its declared repository file stands in.
    if (custom?.source.state._tag === "NotInstalled" && declaration?.origin._tag === "Added") {
      const candidate: DrafterTarget = {
        _tag: "Candidate",
        source: { _tag: "HuggingFace", repository: declaration.origin.repository, commit: declaration.origin.commit },
        files: declaration.target,
      }
      return yield* nativeTarget(candidate)
    }
    const primary = custom?.source.files[0]?.path
    return {
      request: (other: Option.Option<string>) =>
        ({ _tag: "Model", modelId: target.modelId, other }) as unknown as Native.DrafterCandidatesRequest,
      candidate: primary === undefined || !isAbsolute(primary)
        ? Option.none<Native.ModelCandidate>()
        : Option.some<Native.ModelCandidate>({
            source: { _tag: "Local", path: dirname(primary) } as Native.CandidateSource,
            target: custom!.source.files.map(({ path }) => basename(path)),
            projector: Option.none(),
            drafter: Option.none(),
          }),
    }
  })

  const listDrafterCandidates = (target: DrafterTarget, other: Option.Option<string>) => Effect.gen(function* () {
    const { request } = yield* nativeTarget(target)
    const label = Option.getOrElse(other, () => target._tag === "Model" ? target.modelId : target.files[0])
    const entered = Option.map(other, (input) => drafterInput(input, locations.environment))
    const native = yield* icn.drafters(request(entered)).pipe(Effect.mapError(candidateFailure(label)))
    const { catalogModels } = yield* sources.state
    const candidates = native.flatMap((candidate) => Option.toArray(productDrafterCandidate(candidate, catalogModels)))
    yield* Effect.forEach(candidates, (candidate) => candidate.verdict._tag === "Bound"
      ? methods.remember(candidate.reference, candidate.verdict.method)
      : Effect.void, { discard: true })
    return candidates
  })

  /** How ICN's "different drafter" input names a reference it did not list. */
  const otherReference = (reference: DrafterReference): Option.Option<string> => {
    switch (reference._tag) {
      case "HuggingFace": return Option.some(`${reference.repository}@${reference.commit}:${reference.path}`)
      case "Local": return Option.some(reference.path)
      default: return Option.none()
    }
  }

  /** The drafter a choice resolves to; a rejected or unknown reference is refused, never attached. */
  const chooseDrafter = (target: DrafterTarget, choice: DrafterChoice) => Effect.gen(function* () {
    if (choice._tag === "None") return Option.none<DrafterCandidate>()
    const candidates = yield* listDrafterCandidates(target, Option.none())
    if (choice._tag === "Auto") return preselectedDrafterCandidate(candidates)
    const key = drafterReferenceKey(choice.reference)
    const matching = (list: readonly DrafterCandidate[]) =>
      Option.fromNullable(list.find(({ reference }) => drafterReferenceKey(reference) === key))
    const candidate = yield* Option.match(matching(candidates), {
      onSome: Effect.succeed,
      onNone: () => Option.match(otherReference(choice.reference), {
        onNone: () => Effect.fail(new InvalidModelReference({ reference: key, message: "This drafter isn't available for this model." })),
        onSome: (other) => listDrafterCandidates(target, Option.some(other)).pipe(
          Effect.flatMap((listed) => Option.match(Option.orElse(matching(listed), () => Option.fromNullable(
            listed.find(({ category }) => category === "Other"))), {
            onNone: () => Effect.fail(new InvalidModelReference({ reference: other, message: "Magnitude couldn't check this drafter." })),
            onSome: Effect.succeed,
          })),
        ),
      }),
    })
    if (candidate.verdict._tag !== "Bound") {
      return yield* new DrafterNotUsable({
        reference: key,
        reasons: candidate.verdict._tag === "Rejected" ? candidate.verdict.reasons : [],
      })
    }
    return Option.some(candidate)
  })

  const candidateDrafter = (chosen: Option.Option<DrafterCandidate>) => Effect.gen(function* () {
    if (Option.isNone(chosen)) return Option.none<Native.CandidateDrafter>()
    const { reference, verdict, method: known } = chosen.value
    const method = verdict._tag === "Bound"
      ? Option.some(verdict.method)
      : Option.isSome(known) ? known : yield* methods.get(reference)
    return Option.map(method, (value): Native.CandidateDrafter => ({
      method: value,
      reference: nativeDrafterReference(reference),
    }) as Native.CandidateDrafter)
  })

  /** Fetches only the drafter of a model whose target is already on this computer. */
  const startDrafterOnly = (modelId: CustomFormModelId, target: DrafterTarget, chosen: DrafterCandidate) => Effect.gen(function* () {
    if (chosen.onThisComputer || chosen.reference._tag === "Embedded") return false
    const { candidate } = yield* nativeTarget(target)
    const drafter = yield* candidateDrafter(Option.some(chosen))
    if (Option.isNone(candidate) || Option.isNone(drafter)) return false
    const draft = drafterPackage(yield* icn.bundle({ ...candidate.value, drafter }))
    if (Option.isNone(draft)) return false
    yield* downloads.start({ _tag: "Standalone", package: draft.value } as Native.ServableModelBundle, modelId)
    return true
  })

  const startAddedDownload = (
    modelId: CustomFormModelId,
    declaration: CustomModelDeclaration,
    chosen: Option.Option<DrafterCandidate>,
  ) => Effect.gen(function* () {
    if (declaration.origin._tag !== "Added") return false
    // A separate drafter already on this computer, in any location, is used where it is, never downloaded.
    const downloaded = Option.filter(chosen, ({ reference, onThisComputer }) => reference._tag === "Embedded" || !onThisComputer)
    const bundle = yield* icn.bundle({
      source: { _tag: "HuggingFace", repository: declaration.origin.repository, commit: declaration.origin.commit } as Native.CandidateSource,
      target: [...declaration.target],
      projector: Option.map(declaration.projector, (path) => ({ path, sizeBytes: 0 })),
      drafter: yield* candidateDrafter(downloaded),
    })
    yield* downloads.start(bundle, modelId)
    return true
  })

  type AddedDeclaration = CustomModelDeclaration & { readonly origin: Extract<CustomModelDeclaration["origin"], { readonly _tag: "Added" }> }

  /**
   * An added model's version re-resolved at a newer commit: the same version, which its model ID
   * names, with that commit's shards and projector, and a drafter from the same repository moved to
   * that commit when it is still there.
   */
  const resolveAddedAt = (
    modelId: CustomFormModelId,
    declaration: AddedDeclaration,
    commit: string,
    chosen: Option.Option<DrafterCandidate>,
  ) => Effect.gen(function* () {
    const { repository } = declaration.origin
    const native = yield* icn.inspect({ _tag: "HuggingFace", repository, revision: commit })
      .pipe(Effect.mapError(candidateFailure(`${repository}@${commit}`)))
    const version = native.versions.find((candidate) => candidate.modelId === modelId)
    const target = version?.files.map(({ path }) => path) ?? []
    if (!Arr.isNonEmptyReadonlyArray(target) || native.source._tag !== "HuggingFace") {
      return yield* new LocalModelMutationFailed({
        code: "custom_model_update_unavailable",
        message: `${repository} no longer has this version at its newest commit.`,
        retryable: false,
      })
    }
    const source = { _tag: "HuggingFace" as const, repository, commit: native.source.commit }
    const drafter = yield* Option.match(
      Option.filter(chosen, ({ reference }) => reference._tag === "HuggingFace" && reference.repository === repository),
      {
        onNone: () => Effect.succeed(chosen),
        // A drafter the new commit no longer has stays at the commit it was pinned to.
        onSome: ({ reference }) => chooseDrafter(
          { _tag: "Candidate", source, files: target },
          { _tag: "Reference", reference: { ...reference, commit: source.commit } as DrafterReference },
        ).pipe(Effect.orElseSucceed(() => chosen)),
      },
    )
    const resolved: AddedDeclaration = {
      origin: { _tag: "Added", repository, commit: source.commit, pendingCommit: Option.none() },
      target,
      projector: Option.map(native.projector, ({ path }) => path),
    }
    return { declaration: resolved, drafter }
  })

  /** The drafter a model fetches with its files: the attached one, else the persisted choice ICN has not bound yet. */
  const drafterToFetch = (modelId: CustomFormModelId, config: CustomModelsConfig) => Effect.gen(function* () {
    const attached = Option.flatMap(yield* rowFor(modelId), ({ drafter }) => drafter)
    const configured = config.drafters[modelId]
    const { catalogModels } = yield* sources.state
    return Option.orElse(
      Option.map(attached, (drafter): DrafterCandidate => ({
        category: "Other", reference: drafter.reference, name: drafter.name, label: drafter.label,
        sizeBytes: drafter.sizeBytes, lineage: "None", verdict: drafter.verdict, automatic: false, onThisComputer: false,
        method: Option.none(),
      })),
      () => configured?._tag === "Reference" ? Option.some<DrafterCandidate>({
        category: "Other", reference: configured.reference,
        ...drafterReferenceNaming(configured.reference, catalogModels), sizeBytes: 0,
        lineage: "None", verdict: { _tag: "None" }, automatic: false, onThisComputer: false, method: Option.some(configured.method),
      }) : Option.none(),
    )
  })

  /** Changes an added declaration still pending `commit`; a removed model or another update is left alone. */
  const modifyPending = (
    modelId: CustomFormModelId,
    commit: string,
    f: (declaration: AddedDeclaration, config: CustomModelsConfig) => CustomModelsConfig,
  ) => customModels.modify((config) => {
    const current = config.declarations[modelId]
    const pending = current?.origin._tag === "Added" && Option.contains(current.origin.pendingCommit, commit)
    return Effect.succeed([pending, pending ? f(current as AddedDeclaration, config) : config] as const)
  })

  /**
   * Pins a downloaded update: the declaration takes the newer commit with its shards, projector and
   * a moved same-repository drafter, so ICN serves the new copy and removes the one it replaced.
   */
  const finishUpdate = (modelId: CustomFormModelId, declaration: AddedDeclaration, commit: string) => Effect.gen(function* () {
    const config = yield* customModels.get
    const { declaration: pinned, drafter } = yield* resolveAddedAt(modelId, declaration, commit, yield* drafterToFetch(modelId, config))
    const changed = yield* modifyPending(modelId, commit, (_, current) => {
      const configured = current.drafters[modelId]
      return {
        ...current,
        declarations: { ...current.declarations, [modelId]: pinned },
        drafters: configured?._tag === "Reference"
          ? { ...current.drafters, [modelId]: pinnedChoice(configured, drafter) }
          : current.drafters,
      }
    })
    if (changed) yield* settle
  })

  /** Abandons an update that was cancelled or failed: the model stays on its pinned commit and keeps serving. */
  const abandonUpdate = (modelId: CustomFormModelId, commit: string) => modifyPending(modelId, commit, (declaration, config) => ({
    ...config,
    declarations: { ...config.declarations, [modelId]: { ...declaration, origin: { ...declaration.origin, pendingCommit: Option.none() } } },
  }))

  /**
   * Fetches an added model's update at `commit` while it keeps serving its pinned commit, recording
   * the commit as pending once the download is admitted. An update whose files are all here already
   * is pinned at once.
   */
  const startUpdate = (modelId: CustomFormModelId, declaration: AddedDeclaration, commit: string) => Effect.gen(function* () {
    const config = yield* customModels.get
    const resolved = yield* resolveAddedAt(modelId, declaration, commit, yield* drafterToFetch(modelId, config))
    const downloaded = Option.filter(resolved.drafter, ({ reference, onThisComputer }) => reference._tag === "Embedded" || !onThisComputer)
    const bundle = yield* icn.bundle({
      source: { _tag: "HuggingFace", repository: resolved.declaration.origin.repository, commit: resolved.declaration.origin.commit } as Native.CandidateSource,
      target: [...resolved.declaration.target],
      projector: Option.map(resolved.declaration.projector, (path) => ({ path, sizeBytes: 0 })),
      drafter: yield* candidateDrafter(downloaded),
    })
    const download = yield* downloads.start(bundle, modelId)
    yield* customModels.modify((current) => {
      const latest = current.declarations[modelId]
      return Effect.succeed([undefined, latest?.origin._tag !== "Added" ? current : {
        ...current,
        declarations: { ...current.declarations, [modelId]: { ...latest, origin: { ...latest.origin, pendingCommit: Option.some(commit) } } },
      }] as const)
    })
    if (Option.isNone(download)) yield* finishUpdate(modelId, declaration, commit)
  })

  const pendingUpdateLock = yield* Effect.makeSemaphore(1)

  /**
   * Settles every pending update from its download: a finished one is pinned, a cancelled or failed one
   * is abandoned, and one a restart left without a download is fetched again, keeping its bytes.
   */
  const settlePendingUpdates = pendingUpdateLock.withPermits(1)(Effect.gen(function* () {
    const config = yield* customModels.get
    const all = (yield* downloads.get).state.downloads
    yield* Effect.forEach(Object.entries(config.declarations), ([id, declaration]) => {
      if (declaration.origin._tag !== "Added" || Option.isNone(declaration.origin.pendingCommit)) return Effect.void
      const modelId = id as CustomFormModelId
      const added = declaration as AddedDeclaration
      const commit = declaration.origin.pendingCommit.value
      const { repository } = declaration.origin
      const download = all.findLast((candidate) => Option.contains(candidate.forModel, modelId)
        && bundlePackages(candidate.bundle).some((pkg) => packageHoldsFile(pkg, repository, commit, declaration.target[0])))
      const step: Effect.Effect<unknown, unknown> = download === undefined
        ? startUpdate(modelId, added, commit)
        : downloadIsActive(download)
          ? Effect.void
          : download.state._tag === "Completed"
            ? finishUpdate(modelId, added, commit)
            : abandonUpdate(modelId, commit)
      return step.pipe(Effect.catchAll((error) => Effect.logWarning(
        `Unable to settle the update of ${modelId}: ${error instanceof Error ? error.message : String(error)}`,
      )))
    }, { discard: true })
  }))

  const persistDrafter = (modelId: CustomFormModelId, choice: PersistedDrafterChoice, declaration: Option.Option<CustomModelDeclaration>) =>
    customModels.modify((config) => Effect.succeed([undefined, {
      ...config,
      declarations: Option.match(declaration, {
        onNone: () => config.declarations,
        onSome: (value) => ({ ...config.declarations, [modelId]: value }),
      }),
      drafters: { ...config.drafters, [modelId]: choice },
      hidden: config.hidden.filter((hidden) => hidden !== modelId),
    } satisfies CustomModelsConfig] as const))

  const settle = reconciler.push.pipe(Effect.catchAll((error) => Effect.logWarning(error.message)))

  /** After a model is removed, ICN deletes Magnitude's drafter packages no model uses. */
  const pruneDrafters = localModels_.pruneDrafters.pipe(
    Effect.asVoid,
    Effect.catchAll((error) => Effect.logWarning(`Removing unused drafters failed: ${error.message}`)),
  )

  const toMutation = (cause: GeneratedClientError<unknown> | CustomModelMutationError): CustomModelMutationError =>
    cause._tag === "InvalidModelReference" || cause._tag === "ModelRepositoryNotFound" || cause._tag === "ModelSourceUnavailable"
      || cause._tag === "LocalModelMutationFailed" || cause._tag === "ModelVersionNotRunnable" || cause._tag === "DrafterNotUsable"
      ? cause
      : candidateFailure("model")(cause)

  const rowFor = (modelId: ModelId) => localModels.state.pipe(Effect.map((state) => Option.fromNullable(
    [...state.models, ...state.hidden].find((model): model is CustomLocalModel => model._tag === "Custom" && model.modelId === modelId),
  )))

  /** The file a path names once links are followed; a path that can't be resolved names itself. */
  const realPath = (path: string) => fs.realPath(path).pipe(Effect.orElseSucceed(() => path))

  const modelFiles = (modelId: CustomFormModelId) => sources.state.pipe(Effect.map(({ customModels: models, hiddenModels }) =>
    [...models, ...hiddenModels].find(({ id }) => id === modelId)?.source.files.map(({ path }) => path) ?? []))

  /** Imported declarations, other than `except`, whose file is one of `paths`, however each was reached. */
  const importedDeclarationsOf = (config: CustomModelsConfig, paths: readonly string[], except: Option.Option<CustomFormModelId>) =>
    Effect.gen(function* () {
      const files = new Set(yield* Effect.forEach(paths, realPath))
      const imported = Object.entries(config.declarations).flatMap(([id, declaration]) =>
        declaration.origin._tag === "Imported" && !Option.contains(except, id as CustomFormModelId)
          ? [{ id: id as CustomFormModelId, path: declaration.origin.path }]
          : [])
      const matching = yield* Effect.filter(imported, ({ path }) => realPath(path).pipe(Effect.map((file) => files.has(file))))
      return matching.map(({ id }) => id)
    })

  /** A catalog model's ID for one of its aliases: the ID its files have as a found model. */
  const catalogModelForAlias = (modelId: ModelId) => sources.state.pipe(Effect.map(({ catalogModels }) => Option.fromNullable(
    catalogModels.find(({ source }) => Option.exists(source.aliases, (aliases) => aliases.includes(modelId as string)))?.id)))

  const isKnownCatalogModel = (modelId: CatalogFormModelId) => Effect.gen(function* () {
    const { catalogModels } = yield* sources.state
    const state = yield* localModels.state
    return catalogModels.some(({ id }) => id === modelId) || [...state.models, ...state.hidden].some((model) => model.modelId === modelId)
  })

  const modelAcquisition = (modelId: CustomFormModelId) => Effect.gen(function* () {
    const config = yield* customModels.get
    const related = customModelDownloads(modelId, config, (yield* downloads.get).state.downloads)
    return { config, related }
  })

  /** The IDs a catalog model's external files have as found models; hiding them hides that copy. */
  const catalogAliases = (modelId: CatalogFormModelId) => sources.state.pipe(Effect.map(({ catalogModels }) =>
    Option.getOrElse(Option.flatMap(Option.fromNullable(catalogModels.find(({ id }) => id === modelId)), ({ source }) => source.aliases), () => [])
      .flatMap((alias) => Option.toArray(Schema.decodeUnknownOption(CustomFormModelIdSchema)(alias)))))

  /** A catalog model whose files are all in an external location is a found copy: hidden, never deleted. */
  const externalCatalogFiles = (modelId: CatalogFormModelId) => Effect.gen(function* () {
    const state = yield* localModels.state
    const model = [...state.models, ...state.hidden].find((candidate) => candidate.modelId === modelId)
    if (model?._tag !== "Catalog" || !("installation" in model.acquisitionState)) return Option.none<ExternalModelFiles>()
    const { installation } = model.acquisitionState
    if (installation.ownership !== "External" || installation._tag !== "Resolved") return Option.none<ExternalModelFiles>()
    const scanned = yield* locations.list(yield* customModels.get)
    return Option.some<ExternalModelFiles>({
      path: installation.primaryPath,
      location: containingLocation(installation.primaryPath, scanned),
    })
  })

  const validateFolder = (path: string): Effect.Effect<string, ModelLocationError> => Effect.gen(function* () {
    const trimmed = path.trim()
    if (!isAbsolute(trimmed)) return yield* new InvalidDirectoryPath({ path })
    const folder = normalize(trimmed)
    const info = yield* fs.stat(folder).pipe(Effect.mapError((error) =>
      error._tag === "SystemError" && error.reason === "PermissionDenied"
        ? new DirectoryAccessDenied({ path: folder })
        : new DirectoryNotFound({ path: folder })))
    if (info.type !== "Directory") return yield* new PathNotDirectory({ path: folder })
    yield* fs.readDirectory(folder).pipe(Effect.mapError(() => new DirectoryAccessDenied({ path: folder })))
    return folder
  })

  const commands = CustomModelCommands.of({
    searchHuggingFace: ({ query, limit, all }) => icn.search({
      query,
      limit: Option.getOrElse(limit, () => 20),
      includeUnsupported: Option.some(all),
    }).pipe(
      Effect.mapError(candidateFailure(query)),
      Effect.map(({ results, hiddenCount }) => ({
        hiddenCount,
        results: results.flatMap((result) => {
          const repository = Schema.decodeUnknownOption(HuggingFaceRepositoryIdSchema)(result.repository)
          if (Option.isNone(repository)) return []
          return Option.toArray(Schema.decodeUnknownOption(HuggingFaceSearchResultSchema)({
            repository: repository.value,
            displayName: repositoryDisplayName(repository.value),
            uploader: repository.value.split("/")[0],
            ...presentField("variantOf", result.variantOf),
            downloads: Option.getOrElse(result.downloads, () => 0),
            ...presentField("architecture", Option.filter(result.architecture, (value) => value.length > 0)),
            ...presentField("unsupported", Option.some(result.unsupported).pipe(Option.filter((reasons) => reasons.length > 0),
              Option.map((reasons) => productUnsupportedReasons(reasons, "")))),
          }))
        }),
      })),
    ),

    inspectCandidate: inspect,

    listDrafterCandidates,

    addHuggingFaceModel: ({ repository, revision, version, drafter, force }) => Effect.gen(function* () {
      const reference = Option.match(version, { onNone: () => repository, onSome: (quant) => `${repository}:${quant}` })
      const native = yield* icn.inspect({ _tag: "HuggingFace", repository, revision: Option.getOrElse(revision, () => "main") })
        .pipe(Effect.mapError(candidateFailure(reference)))
      const inspection = yield* productInspection(native, { file: Option.none(), quant: version }, reference)
      if (Option.isSome(inspection.unsupportedArchitecture)) {
        return yield* new ModelVersionNotRunnable({
          reference,
          version: Option.getOrElse(inspection.selectedVersion, () => repository),
          verdict: { _tag: "Unsupported", reasons: inspection.unsupportedArchitecture.value },
        })
      }
      const selected = inspection.versions.find(({ label }) => Option.contains(inspection.selectedVersion, label))
      if (selected === undefined) {
        return yield* new InvalidModelReference({ reference, message: "Name a version: owner/repository:QUANT." })
      }
      const assessment = Option.getOrUndefined(selected.assessment)
      if (assessment?._tag === "Unsupported") {
        return yield* new ModelVersionNotRunnable({
          reference, version: selected.label, verdict: { _tag: "Unsupported", reasons: assessment.reasons },
        })
      }
      if (assessment?._tag === "DoesNotFit" && !force) {
        return yield* new ModelVersionNotRunnable({
          reference, version: selected.label, verdict: { _tag: "DoesNotFit", deficitBytes: assessment.deficitBytes },
        })
      }
      const source = inspection.source
      if (source._tag !== "HuggingFace") return yield* new InvalidModelReference({ reference, message: "Not a Hugging Face repository." })
      // A version that is exactly a catalog model is that model, with the catalog's drafter: install it, never declare it.
      if (Option.isSome(selected.catalogModel)) {
        const catalogModel = selected.catalogModel.value
        if (drafter._tag !== "Auto") {
          return yield* new InvalidModelReference({
            reference,
            message: `${reference} is the catalog model ${catalogModel}, which comes with its drafter.`,
          })
        }
        if (yield* catalogSync.installed(catalogModel)) {
          return { _tag: "Catalog" as const, modelId: catalogModel, version: selected.label, outcome: "AlreadyOnThisComputer" as const }
        }
        yield* catalogSync.sync(catalogModel)
        return { _tag: "Catalog" as const, modelId: catalogModel, version: selected.label, outcome: "Started" as const }
      }
      const existing = Option.flatMap(selected.onThisComputer, Schema.decodeUnknownOption(CustomFormModelIdSchema))
      const modelId = Option.getOrElse(existing, () => selected.modelId)
      const target: DrafterTarget = Option.isSome(existing)
        ? { _tag: "Model", modelId }
        : { _tag: "Candidate", source, files: selected.files }
      const chosen = yield* chooseDrafter(target, drafter)
      const declaration: Option.Option<CustomModelDeclaration> = Option.isSome(existing)
        ? Option.none()
        : Option.some({
            origin: { _tag: "Added", repository: source.repository, commit: source.commit, pendingCommit: Option.none() },
            target: selected.files,
            projector: Option.map(inspection.projector, ({ path }) => path),
          })
      // Auto stays Auto, so the model follows automatic choice (a catalog drafter by lineage included); only a
      // drafter the user names is pinned.
      yield* persistDrafter(modelId, persistedChoice(drafter, chosen, Option.none()), declaration)
      yield* settle
      const drafterDownloadStarted = Option.isSome(declaration)
        ? yield* startAddedDownload(modelId, declaration.value, chosen).pipe(Effect.as(false))
        : yield* Option.match(chosen, { onNone: () => Effect.succeed(false), onSome: (value) => startDrafterOnly(modelId, target, value) })
      return {
        _tag: "Custom" as const,
        modelId,
        version: selected.label,
        drafter: chosen,
        outcome: Option.isSome(existing) ? "AlreadyOnThisComputer" as const : "Started" as const,
        drafterDownloadStarted,
      }
    }).pipe(Effect.mapError(toMutation)),

    importModelFile: (path, drafter) => Effect.gen(function* () {
      const reference = path
      const parsed = yield* parse(path.startsWith("~") || isAbsolute(path) ? path : `./${path}`)
      if (parsed._tag !== "Local") return yield* new InvalidModelReference({ reference, message: "Choose a GGUF file." })
      const info = yield* fs.stat(parsed.path).pipe(Effect.option)
      if (Option.exists(info, ({ type }) => type === "Directory")) {
        return yield* new InvalidModelReference({ reference, message: `${parsed.path} is a folder. Choose a GGUF file, or add the folder as a model location.` })
      }
      const native = yield* icn.inspect({ _tag: "Local", path: parsed.path }).pipe(Effect.mapError(candidateFailure(reference)))
      const inspection = yield* productInspection(native, { file: Option.none(), quant: Option.none() }, reference)
      const version = inspection.versions[0]
      const assessment = version === undefined ? undefined : Option.getOrUndefined(version.assessment)
      const unsupported = Option.orElse(
        inspection.unsupportedArchitecture,
        () => assessment?._tag === "Unsupported" ? Option.some(assessment.reasons) : Option.none(),
      )
      if (Option.isSome(unsupported)) {
        return yield* new ModelVersionNotRunnable({
          reference,
          version: basename(parsed.path),
          verdict: { _tag: "Unsupported", reasons: unsupported.value },
        })
      }
      if (version === undefined) return yield* new InvalidModelReference({ reference, message: "Magnitude can't run this file." })
      // The same bytes are one model: a file whose bytes are already a model on this computer is that model.
      const existing = Option.flatMap(version.onThisComputer, Schema.decodeUnknownOption(CustomFormModelIdSchema))
      const modelId = Option.getOrElse(existing, () => version.modelId)
      const identity = parseModelId(version.modelId)
      const target: DrafterTarget = Option.isSome(existing)
        ? { _tag: "Model", modelId }
        : { _tag: "Candidate", source: inspection.source, files: version.files }
      // A model already here keeps its drafter unless the import names one or turns it off.
      const current = (yield* customModels.get).drafters[modelId]
      const requested = Option.isSome(existing) && drafter._tag === "Auto" && current !== undefined ? current : drafter
      const chosen = yield* chooseDrafter(target, requested)
      const recommended = Option.flatMap(
        Option.isSome(existing) ? yield* rowFor(modelId) : Option.none(),
        ({ recommendedDrafter }) => Option.map(recommendedDrafter, ({ reference }) => reference),
      )
      // A file without a Hugging Face identity is recorded where it is: a new model, or one more location of the
      // model that already has its bytes, which keeps that model's identity and origin. A file with a Hugging Face
      // identity is found where it is; only its drafter is recorded. Links are followed first: a file already
      // listed or declared, through a link or its own path, is recorded once.
      const file = yield* realPath(parsed.path)
      const listed = (yield* Effect.forEach(Option.isSome(existing) ? yield* modelFiles(modelId) : [], realPath)).includes(file)
      const declared = (yield* importedDeclarationsOf(yield* customModels.get, [file], Option.none())).length > 0
      const declaration: CustomModelDeclaration | undefined = identity._tag === "Local" && !listed && !declared
        ? {
            origin: { _tag: "Imported", path: parsed.path, headerDigest: identity.headerDigest },
            target: [parsed.path],
            projector: Option.map(inspection.projector, ({ path: projector }) => projector),
          }
        : undefined
      yield* customModels.modify((config) => Effect.succeed([undefined, {
        ...config,
        declarations: declaration === undefined || config.declarations[version.modelId] !== undefined
          ? config.declarations
          : { ...config.declarations, [version.modelId]: declaration },
        drafters: { ...config.drafters, [modelId]: persistedChoice(requested, chosen, recommended) },
        hidden: config.hidden.filter((hidden) => hidden !== modelId),
      } satisfies CustomModelsConfig] as const))
      yield* settle
      const drafterDownloadStarted = yield* Option.match(chosen, {
        onNone: () => Effect.succeed(false),
        onSome: (value) => startDrafterOnly(modelId, target, value),
      })
      return {
        modelId,
        drafter: chosen,
        outcome: Option.isSome(existing) ? "AlreadyOnThisComputer" as const : "Started" as const,
        drafterDownloadStarted,
      }
    }).pipe(Effect.mapError(toMutation)),

    setModelDrafter: (modelId, drafter) => Effect.gen(function* () {
      const row = yield* rowFor(modelId)
      if (Option.isNone(row)) return yield* notFound(modelId)
      const chosen = yield* chooseDrafter({ _tag: "Model", modelId }, drafter)
      const recommended = Option.map(row.value.recommendedDrafter, ({ reference }) => reference)
      yield* customModels.modify((config) => Effect.succeed([undefined, {
        ...config,
        drafters: { ...config.drafters, [modelId]: persistedChoice(drafter, chosen, recommended) },
      }] as const))
      // Changing or turning off a drafter only changes the choice: the drafter stays on disk, so turning it back on
      // downloads nothing. Unused drafters go when a model is removed.
      yield* settle
      const drafterDownloadStarted = yield* Option.match(chosen, {
        onNone: () => Effect.succeed(false),
        onSome: (value) => startDrafterOnly(modelId, { _tag: "Model", modelId }, value),
      })
      return { drafter: chosen, drafterDownloadStarted }
    }).pipe(Effect.mapError(toMutation)),

    hide: (requested) => Effect.gen(function* () {
      // An alias names its catalog model. A catalog model found only in an external location is hidden by its
      // files' found-model IDs, which ICN then stops attributing to the catalog model; hiding it again changes nothing.
      const modelId: ModelId = Option.getOrElse(yield* catalogModelForAlias(requested), () => requested)
      const known = Schema.is(CatalogFormModelIdSchema)(modelId)
        ? yield* isKnownCatalogModel(modelId)
        : Option.isSome(yield* rowFor(modelId)) || (yield* customModels.get).declarations[modelId as CustomFormModelId] !== undefined
      if (!known) return yield* notFound(modelId)
      const hiddenIds = Schema.is(CatalogFormModelIdSchema)(modelId)
        ? yield* Effect.gen(function* () {
            const aliases = yield* catalogAliases(modelId)
            const { hidden } = yield* customModels.get
            const alreadyHidden = aliases.length > 0 && aliases.every((alias) => hidden.includes(alias))
            return alreadyHidden || Option.isSome(yield* externalCatalogFiles(modelId))
              ? aliases
              : yield* catalogHideRefusal(modelId)
          })
        : [modelId]
      const declaration = Schema.is(CustomFormModelIdSchema)(modelId) ? (yield* customModels.get).declarations[modelId] : undefined
      if (declaration !== undefined) return yield* managedHideRefusal(modelId, declaration.origin._tag)
      yield* customModels.modify((config) => Effect.succeed([undefined, {
        ...config,
        hidden: [...config.hidden, ...hiddenIds.filter((id) => !config.hidden.includes(id))],
      }] as const))
      yield* settle
    }),

    externalCatalogFiles,

    unhide: (modelId) => Effect.gen(function* () {
      const shownIds: readonly ModelId[] = Schema.is(CatalogFormModelIdSchema)(modelId) ? yield* catalogAliases(modelId) : [modelId]
      yield* customModels.modify((config) => Effect.succeed([undefined, {
        ...config,
        hidden: config.hidden.filter((hidden) => !shownIds.includes(hidden)),
      }] as const))
      yield* settle
    }),

    addFolder: (path) => validateFolder(path).pipe(
      Effect.flatMap((folder) => customModels.modify((config) => Effect.succeed([undefined, config.folders.includes(folder)
        ? config
        : { ...config, folders: [...config.folders, folder] }] as const))),
      Effect.zipRight(settle),
    ),

    removeFolder: (path) => Effect.gen(function* () {
      const trimmed = path.trim()
      if (!isAbsolute(trimmed)) return yield* new InvalidDirectoryPath({ path })
      const folder = normalize(trimmed)
      yield* customModels.modify((config) => Effect.succeed([undefined, {
        ...config,
        folders: config.folders.filter((existing) => existing !== folder && existing !== trimmed),
      }] as const))
      yield* settle
    }),

    setLocationEnabled: (kind, enabled) => customModels.modify((config) => Effect.succeed([undefined, {
      ...config,
      disabledLocations: enabled
        ? config.disabledLocations.filter((disabled) => disabled !== kind)
        : config.disabledLocations.includes(kind) ? config.disabledLocations : [...config.disabledLocations, kind],
    }] as const)).pipe(Effect.zipRight(settle)),

    listLocations: customModels.get.pipe(Effect.flatMap(locations.list)),

    remove: (modelId) => Effect.gen(function* () {
      const { config, related } = yield* modelAcquisition(modelId)
      const declaration = config.declarations[modelId]
      // A declaration an earlier import left for a file this model lists, reached through a link, goes with it.
      const duplicates = yield* importedDeclarationsOf(config, yield* modelFiles(modelId), Option.some(modelId))
      const forgetDuplicates = (current: CustomModelsConfig): CustomModelsConfig => duplicates.reduce((next, id) => ({
        ...next,
        declarations: withoutKey(next.declarations, id),
        drafters: withoutKey(next.drafters, id),
      }), current)
      if (declaration === undefined) {
        if (Option.isNone(yield* rowFor(modelId))) return yield* notFound(modelId)
        yield* customModels.modify((current) => Effect.succeed([undefined, forgetDuplicates(current.hidden.includes(modelId)
          ? current
          : { ...current, hidden: [...current.hidden, modelId] })] as const))
        yield* settle
        return {}
      }
      yield* Effect.forEach(related.filter(({ download }) => downloadIsActive(download)), ({ download }) =>
        downloads.cancel(download.id).pipe(Effect.ignore), { discard: true })
      if (declaration.origin._tag === "Added") {
        // ICN deletes only Magnitude's copy; the same bytes elsewhere remain and the model stays as found.
        yield* localModels_.removeFiles(modelId).pipe(
          Effect.catchIf((error) => error._tag === "GeneratedClientRemoteError" && error.status === 404, () => Effect.void),
          Effect.mapError(commandFailed("remove")),
        )
      }
      yield* customModels.modify((current) => Effect.succeed([undefined, forgetDuplicates({
        ...current,
        declarations: withoutKey(current.declarations, modelId),
        drafters: withoutKey(current.drafters, modelId),
      })] as const))
      yield* settle
      yield* pruneDrafters
      return {}
    }),

    sync: (modelId) => Effect.gen(function* () {
      const row = yield* rowFor(modelId)
      const { config, related } = yield* modelAcquisition(modelId)
      const declaration = config.declarations[modelId]
      if (Option.isNone(row) && declaration === undefined) return yield* notFound(modelId)
      const attached = Option.flatMap(row, ({ drafter }) => drafter)
      // Until ICN has bound a configured drafter (just after a restart) the persisted choice names it.
      const configured = config.drafters[modelId]
      const sourceState = yield* sources.state
      const chosen = Option.orElse(
        Option.map(attached, (drafter): DrafterCandidate => ({
          category: "Other", reference: drafter.reference, name: drafter.name, label: drafter.label,
          sizeBytes: drafter.sizeBytes, lineage: "None", verdict: drafter.verdict, automatic: false, onThisComputer: false,
          method: Option.none(),
        })),
        () => configured?._tag === "Reference" ? Option.some<DrafterCandidate>({
          category: "Other", reference: configured.reference,
          ...drafterReferenceNaming(configured.reference, sourceState.catalogModels), sizeBytes: 0,
          lineage: "None", verdict: { _tag: "None" }, automatic: false, onThisComputer: false, method: Option.some(configured.method),
        }) : Option.none(),
      )
      const acquisition = Option.map(row, ({ acquisitionState }) => acquisitionState._tag)
      const targetActive = Option.exists(currentDownload(related, "Target"), downloadIsActive)
      const targetMissing = declaration?.origin._tag === "Added" && !targetActive
        && Option.match(acquisition, { onNone: () => true, onSome: (tag) => tag === "NotInstalled" || tag === "InstallFailed" })
      const current = { outcome: "AlreadyCurrent" as LocalModelSyncOutcome }
      if (targetMissing && declaration !== undefined) {
        return (yield* startAddedDownload(modelId, declaration, chosen)) ? { outcome: "Installing" as const } : current
      }
      const updateCommit = Option.flatMap(
        Option.fromNullable(sourceState.customModels.find(({ id }) => id === modelId)),
        ({ source }) => source.updateCommit,
      )
      if (declaration?.origin._tag === "Added" && Option.isSome(updateCommit) && !targetActive
        && Option.exists(acquisition, (tag) => tag === "UpdateAvailable" || tag === "UpdateFailed")) {
        yield* pendingUpdateLock.withPermits(1)(startUpdate(modelId, declaration as AddedDeclaration, updateCommit.value))
        return { outcome: "Updating" as const }
      }
      const drafterUpdate = Option.exists(attached, ({ updateAvailable }) => updateAvailable)
      const drafterMissing = Option.exists(attached, ({ availability }) => availability._tag === "DownloadFailed")
      const drafterActive = Option.exists(currentDownload(related, "Drafter"), downloadIsActive)
      if ((!drafterUpdate && !drafterMissing) || drafterActive || targetActive || Option.isNone(chosen)) return current
      const started = yield* startDrafterOnly(modelId, { _tag: "Model", modelId }, chosen.value)
      return started ? { outcome: drafterUpdate ? "Updating" as const : "DownloadingDrafter" as const } : current
    }).pipe(Effect.mapError((cause) => cause._tag === "LocalModelMutationFailed" ? cause : commandFailed("sync")(
      cause as GeneratedClientError<unknown> | ModelCandidateError,
    ))),

    cancelSync: (modelId) => modelAcquisition(modelId).pipe(
      Effect.flatMap(({ related }) => Effect.forEach(related.filter(({ download }) => downloadIsActive(download)), ({ download }) =>
        downloads.cancel(download.id), { discard: true })),
      Effect.mapError(commandFailed("cancel")),
      Effect.as({}),
    ),

    acknowledgeSyncFailure: (modelId) => modelAcquisition(modelId).pipe(
      Effect.flatMap(({ related }) => Effect.forEach(
        related.filter(({ download }) => downloadHasUnacknowledgedFailure(download)),
        ({ download }) => downloads.acknowledgeFailure(download.id),
        { discard: true },
      )),
      Effect.mapError(commandFailed("acknowledge")),
      Effect.as({}),
    ),
  })

  // Downloads do not outlive the service; every part a restart interrupted resumes, keeping its bytes:
  // an added model's target (with its drafter), and any model's chosen drafter that hasn't arrived.
  // A push first relearns the drafter methods its bundle needs.
  yield* Effect.gen(function* () {
    yield* reconciler.push.pipe(Effect.ignore)
    const state = yield* localModels.state.pipe(
      Effect.filterOrFail((current) => current.preparation.discovery.complete, () => "discovering" as const),
      Effect.retry(Schedule.spaced("1 second")),
    )
    const interrupted = state.models.filter((model): model is CustomLocalModel => model._tag === "Custom")
      .filter(interruptedByRestart)
    yield* Effect.forEach(interrupted, (model) => commands.sync(model.modelId as CustomFormModelId).pipe(
      Effect.catchAll((error) => Effect.logWarning(`Unable to resume the download of ${model.modelId}: ${error.message}`)),
    ), { discard: true })
    // An update a restart interrupted resumes, one that finished or ended meanwhile settles, and from
    // then on each update settles as its download changes.
    yield* settlePendingUpdates
    yield* Stream.merge(downloads.changes, customModels.changes).pipe(Stream.runForEach(() => settlePendingUpdates))
  }).pipe(Effect.forkScoped)
  return commands
}))
