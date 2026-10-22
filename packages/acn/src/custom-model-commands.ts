import * as FileSystem from "@effect/platform/FileSystem"
import { Context, Effect, Either, Layer, Option, Schema, Stream } from "effect"
import { dirname, basename, isAbsolute, normalize, sep } from "node:path"
import {
  CandidateInspectionSchema,
  canonicalLocalModelId,
  CatalogFormModelIdSchema,
  CustomFormModelIdSchema,
  DirectoryAccessDenied,
  DirectoryNotFound,
  DrafterNotUsable,
  HuggingFaceFormModelIdSchema,
  HuggingFaceRepositoryIdSchema,
  HuggingFaceSearchResultSchema,
  InvalidDirectoryPath,
  InvalidModelReference,
  LocalModelAssessmentSchema,
  LocalModelMutationFailed,
  ModelRepositoryNotFound,
  ModelSourceUnavailable,
  HuggingFaceAccessRequired,
  ModelVersionNotRunnable,
  PathNotDirectory,
  preselectedDrafterCandidate,
  sameModelId,
  type AutomaticModelLocationKind,
  type CandidateInspection,
  type CandidateInspectionProgress,
  type CandidateInspectionUpdate,
  type CatalogFormModelId,
  type CustomFormModelId,
  type CustomLocalModel,
  type AddedModel,
  type CustomModelMutationError,
  type DrafterCandidate,
  type DrafterChoice,
  type DrafterReference,
  type DrafterTarget,
  type HuggingFaceFormModelId,
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
  IcnCandidates,
  IcnDownloads,
  IcnLocalModels,
  IcnModelAssessments,
  packageHoldsFile,
} from "@magnitudedev/icn"
import type * as Native from "@magnitudedev/icn-protocol/schemas"
import type { GeneratedClientError } from "@magnitudedev/openapi-effect/client-runtime"
import type { AddedModelSetting, CustomModelsConfig } from "@magnitudedev/storage"
import { CatalogModelSync } from "./catalog-model-sync"
import {
  addedModel,
  configuredDrafter,
  CustomModels,
  drafterSettingReference,
  isHidden,
  withCanonicalReferences,
  withDrafterChoice,
  withLocationEnabled,
} from "./custom-models"
import { CustomModelReconciler } from "./custom-model-reconciler"
import {
  addedTargetPackage,
  currentDownload,
  customModelDownloads,
  downloadHasUnacknowledgedFailure,
  downloadIsActive,
  type RelatedDownload,
} from "./custom-model-downloads"
import {
  drafterReferenceKey,
  presentField,
  productDrafterCandidate,
  productDrafterReference,
} from "./drafter-references"
import { LocalModelSources } from "./local-model-sources"
import { LocalModels, projectAssessment } from "./local-models"
import { ModelLocations } from "./model-locations"
import { drafterInput, parseModelReference, repositoryDisplayName } from "./model-references"
import { HuggingFaceAccount } from "./hugging-face-account"
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

const catalogHideRefusal = (modelId: ModelId) => new LocalModelMutationFailed({
  code: "catalog_model_not_hideable",
  message: `${modelId} is a catalog model. Remove it with \`magnitude catalog remove ${modelId}\` instead.`,
  retryable: false,
})

const managedHideRefusal = (modelId: ModelId) => new LocalModelMutationFailed({
  code: "managed_model_not_hideable",
  message: `${modelId} was added to Magnitude, so it can't be hidden. Remove it with \`magnitude catalog remove ${modelId}\` instead.`,
  retryable: false,
})

/** The location holding a path: the one with the deepest scanned directory that contains it. */
const containingLocation = (path: string, locations: readonly ModelLocation[]): Option.Option<ModelLocationKind> =>
  Option.fromNullable(locations
    .flatMap((location) => location.paths.map((directory) => ({ kind: location.kind, directory: directory.path })))
    .filter(({ directory }) => path === directory || path.startsWith(`${directory.replace(/[\\/]+$/, "")}${sep}`))
    .sort((left, right) => right.directory.length - left.directory.length)[0]?.kind)

/** Custom-model commands: finding and adding models, drafters, hiding, and model locations. */
export interface DrafterChoices {
  readonly candidates: readonly DrafterCandidate[]
  readonly entered: Option.Option<DrafterCandidate>
}

export interface CustomModelCommandsApi {
  readonly searchHuggingFace: (request: {
    readonly query: string
    readonly limit: Option.Option<number>
  }) => Effect.Effect<HuggingFaceSearchResponse, ModelCandidateError>
  readonly inspectCandidate: (reference: string) => Effect.Effect<CandidateInspection, ModelCandidateError>
  /** The same inspection as it goes: its progress in order, then the inspection itself. */
  readonly inspectCandidateWithProgress: (reference: string) => Stream.Stream<CandidateInspectionUpdate, ModelCandidateError>
  /** The drafter choices, and the listed candidate a different drafter (`other`) resolves to. */
  readonly listDrafterCandidates: (
    target: DrafterTarget,
    other: Option.Option<string>,
  ) => Effect.Effect<DrafterChoices, ModelCandidateError>
  readonly addHuggingFaceModel: (request: {
    readonly repository: HuggingFaceRepositoryId
    readonly revision: Option.Option<string>
    readonly version: Option.Option<string>
    readonly drafter: DrafterChoice
    readonly force: boolean
  }) => Effect.Effect<AddedModel, CustomModelMutationError>
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

/** The product message for a gated repository while Magnitude isn't signed in to Hugging Face. */
export const signInRequiredMessage = "Sign in to Hugging Face to use this model."

/** The product message for a gated repository whose terms the signed-in account hasn't accepted. */
export const termsRequiredMessage = "Accept this model's terms on Hugging Face, then try again. "
  + "Some models also need the author's approval."

/** The product message for a repository Hugging Face won't show: missing, private or gated. */
export const repositoryNotFoundMessage = (signedIn: boolean): string => signedIn
  ? "Repository not found, or your Hugging Face account doesn't have access to it. Check the name."
  : "Repository not found, or it's private or gated. Check the name, or sign in to Hugging Face if you have access."

/** The Hugging Face repository a reference names: a link, owner/repo, owner/repo:QUANT or owner/repo@commit. */
export const referencedRepository = (reference: string): string =>
  /^(?:https?:\/\/(?:www\.)?(?:huggingface\.co|hf\.co)\/)?([\w.-]+\/[\w.-]+)/.exec(reference.trim())?.[1] ?? reference

/** The product message for a repository that lists no GGUF file. */
export const noGgufFilesMessage = "This repository has no GGUF files Magnitude can use."

/** The product message for a file whose contents Magnitude couldn't read. */
export const unreadableFileMessage = "Magnitude couldn't read this file. It may be incomplete, damaged, "
  + "or use a format Magnitude doesn't support yet."

/** The ICN error type: a failure reported after an inspection stream began has no status of its own. */
const remoteType = (cause: GeneratedClientError<unknown>): string | undefined => {
  if (cause._tag !== "GeneratedClientRemoteError") return undefined
  const type = (cause.body as { readonly error?: { readonly type?: unknown } } | undefined)?.error?.type
  return typeof type === "string" ? type : undefined
}

const remoteCode = (cause: GeneratedClientError<unknown>): string | undefined => {
  if (cause._tag !== "GeneratedClientRemoteError") return undefined
  const code = (cause.body as { readonly error?: { readonly code?: unknown } } | undefined)?.error?.code
  return typeof code === "string" ? code : undefined
}

/**
 * A rejected reference is the caller's to fix. A gated repository, one Hugging Face won't show
 * (worded by whether Magnitude is signed in to Hugging Face), one without GGUF files and a file that
 * can't be read get product messages; other Hugging Face and file failures keep the inference
 * service's own message. Only a connection that never reached it is a service reachability
 * failure, and a response that doesn't match the contract is never presented as one.
 */
export const candidateFailure = (reference: string, signedIn: boolean) => (cause: GeneratedClientError<unknown>): ModelCandidateError => {
  switch (cause._tag) {
    case "GeneratedClientRemoteError": {
      if (cause.status === 400 || cause.status === 404) return new InvalidModelReference({ reference, message: remoteMessage(cause) })
      switch (remoteCode(cause)) {
        case "hugging_face_sign_in_required":
          return new HuggingFaceAccessRequired({ repository: referencedRepository(reference), access: "SignIn", message: signInRequiredMessage })
        case "hugging_face_terms_required":
          return new HuggingFaceAccessRequired({ repository: referencedRepository(reference), access: "AcceptTerms", message: termsRequiredMessage })
        case "hugging_face_repository_not_found":
          return new ModelRepositoryNotFound({ reference, message: repositoryNotFoundMessage(signedIn) })
        case "hugging_face_repository_without_gguf":
          return new ModelSourceUnavailable({ reason: "Unavailable", message: noGgufFilesMessage, retryable: false })
        case "integrity_failed":
          return new ModelSourceUnavailable({ reason: "Unreadable", message: unreadableFileMessage, retryable: false })
        default: {
          const retryable = cause.status === 409 || cause.status >= 500 || remoteType(cause) === "server_error"
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
    || cause._tag === "HuggingFaceAccessRequired"
    ? new LocalModelMutationFailed({ code: `custom_model_${operation}_failed`, message: cause.message, retryable: true })
    : new LocalModelMutationFailed({
        code: `custom_model_${operation}_failed`,
        message: remoteMessage(cause),
        retryable: cause._tag !== "GeneratedClientRemoteError" || cause.status >= 500,
      })

const withoutKey = <K extends string, V>(record: Readonly<Partial<Record<K, V>>>, key: K): Partial<Record<K, V>> => {
  const next = { ...record }
  delete next[key]
  return next
}

/**
 * The drafter choice persisted for a model: Auto stays Auto, as does picking the recommended drafter (by the
 * candidate the choice resolved to, so the catalog drafter's file named by its Hugging Face path counts), so a
 * later catalog drafter still applies; any other reference is pinned as the candidate it resolved to, when the engine
 * binds it, and the engine works out its method again wherever it is used.
 */
const persistedChoice = (
  requested: DrafterChoice,
  chosen: Option.Option<DrafterCandidate>,
  recommended: Option.Option<DrafterReference>,
): DrafterChoice => {
  if (requested._tag === "None") return { _tag: "None" }
  const recommendedPicked = requested._tag === "Reference" && Option.exists(recommended,
    (reference) => Option.exists(chosen, (candidate) => drafterReferenceKey(candidate.reference) === drafterReferenceKey(reference)))
  if (requested._tag === "Auto" || recommendedPicked) return { _tag: "Auto" }
  return Option.match(Option.filter(chosen, ({ verdict }) => verdict._tag === "Bound"), {
    onNone: (): DrafterChoice => ({ _tag: "Auto" }),
    onSome: ({ reference }): DrafterChoice => ({ _tag: "Reference", reference }),
  })
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
  | ModelLocations
  | CatalogModelSync
  | HuggingFaceAccount
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
  const locations = yield* ModelLocations
  const catalogSync = yield* CatalogModelSync
  const fs = yield* FileSystem.FileSystem
  const huggingFace = yield* HuggingFaceAccount

  // The assessments a candidate's versions are projected against: their environment and the
  // machine's measurement progress, which a pending speed waits on.
  const assessmentsSnapshot = assessments.get.pipe(Effect.map(({ state }) => state))

  const candidateFailed = (reference: string) => (cause: GeneratedClientError<unknown>) => huggingFace.account.pipe(
    Effect.flatMap((account) => Effect.fail(candidateFailure(reference, account._tag === "SignedIn")(cause))),
  )

  const parse = (reference: string) => Either.match(parseModelReference(reference, locations.environment), {
    onLeft: (message) => Effect.fail(new InvalidModelReference({ reference, message })),
    onRight: Effect.succeed,
  })

  const productInspection = (
    native: Native.CandidateInspection,
    selection: { readonly file: Option.Option<string>; readonly quant: Option.Option<string> },
    reference: string,
  ) => Effect.gen(function* () {
    const snapshot = yield* assessmentsSnapshot
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
        assessment: Option.map(version.assessment, (assessment) => projectAssessment(snapshot, assessment)),
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
    ).pipe(Effect.catchAll(candidateFailed(reference)))
    return yield* productInspection(
      native,
      parsed._tag === "HuggingFace" ? parsed : { file: Option.none(), quant: Option.none() },
      reference,
    )
  })

  const productProgress = (snapshot: Native.ModelAssessmentsSnapshot, native: Native.CandidateInspectionProgress): CandidateInspectionProgress => ({
    phase: native.phase._tag === "CheckingDrafters"
      ? { _tag: "CheckingDrafters", version: native.phase.version, completed: native.phase.completed, total: native.phase.total }
      : { _tag: native.phase._tag, completed: native.phase.completed, total: native.phase.total },
    versions: native.versions.map((version) => ({
      label: version.label,
      sizeBytes: version.sizeBytes,
      assessment: Option.map(version.assessment, (assessment) => projectAssessment(snapshot, assessment)),
    })),
  })

  const inspectWithProgress = (reference: string) => Stream.unwrap(Effect.gen(function* () {
    const parsed = yield* parse(reference)
    const snapshot = yield* assessmentsSnapshot
    return icn.inspectWithProgress(parsed._tag === "HuggingFace"
      ? { _tag: "HuggingFace", repository: parsed.repository, revision: parsed.revision }
      : { _tag: "Local", path: parsed.path },
    ).pipe(
      Stream.catchAll((cause) => Stream.fromEffect(candidateFailed(reference)(cause))),
      Stream.mapEffect((event): Effect.Effect<CandidateInspectionUpdate, ModelCandidateError> => event._tag === "Progress"
        ? Effect.succeed({ _tag: "Inspecting", progress: productProgress(snapshot, event.progress) })
        : productInspection(
          event.inspection,
          parsed._tag === "HuggingFace" ? parsed : { file: Option.none(), quant: Option.none() },
          reference,
        ).pipe(Effect.map((inspection) => ({ _tag: "Inspected", inspection })))),
    )
  }))

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
    // An added model still downloading has no files yet; its repository files at its pinned commit stand in.
    if (custom?.source.state._tag === "NotInstalled" && custom.source.origin._tag === "Added") {
      const candidate: Native.ModelCandidate = {
        source: { _tag: "HuggingFace", repository: custom.source.origin.repository, commit: custom.source.origin.commit } as Native.CandidateSource,
        target: custom.source.files.map(({ path }) => path),
        projector: Option.none(),
        drafter: Option.none(),
      }
      return {
        request: (other: Option.Option<string>) =>
          ({ _tag: "Candidate", candidate, other }) as Native.DrafterCandidatesRequest,
        candidate: Option.some(candidate),
      }
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

  const listNativeDrafterCandidates = (target: DrafterTarget, other: Option.Option<string>) => Effect.gen(function* () {
    const { request } = yield* nativeTarget(target)
    const label = Option.getOrElse(other, () => target._tag === "Model" ? target.modelId : target.files[0])
    const entered = Option.map(other, (input) => drafterInput(input, locations.environment))
    const native = yield* icn.drafters(request(entered)).pipe(Effect.catchAll(candidateFailed(label)))
    const { catalogModels } = yield* sources.state
    const candidates = native.candidates.flatMap((candidate) => Option.toArray(productDrafterCandidate(candidate, catalogModels)))
    const resolved = Option.flatMap(native.entered, productDrafterReference)
    return {
      candidates,
      entered: Option.flatMap(resolved, (reference) => Option.fromNullable(
        candidates.find((candidate) => drafterReferenceKey(candidate.reference) === drafterReferenceKey(reference)))),
    }
  })

  /** How ICN's "different drafter" input names a reference it did not list. */
  const otherReference = (reference: DrafterReference): Option.Option<string> => {
    switch (reference._tag) {
      case "HuggingFace": return Option.some(`${reference.repository}@${reference.commit}:${reference.path}`)
      case "Local": return Option.some(reference.path)
      default: return Option.none()
    }
  }

  /**
   * A model's drafter choices: ICN's list, then the drafter the user pinned when ICN doesn't list it (one entered by
   * hand), checked again as a different drafter so it shows as Other with its verdict. A pinned file that resolves to a
   * listed candidate adds no row, and one that can't be checked now is left out rather than failing the list.
   */
  const withPinnedDrafter = (modelId: ModelId, target: DrafterTarget, candidates: readonly DrafterCandidate[]) => Effect.gen(function* () {
    const pinned = Option.flatMap(configuredDrafter(yield* customModels.get, modelId), drafterSettingReference)
    if (Option.isNone(pinned)) return candidates
    const listed = new Set(candidates.map(({ reference }) => drafterReferenceKey(reference)))
    if (listed.has(drafterReferenceKey(pinned.value))) return candidates
    const entered = yield* Option.match(otherReference(pinned.value), {
      onNone: () => Effect.succeed(Option.none<DrafterCandidate>()),
      onSome: (input) => listNativeDrafterCandidates(target, Option.some(input)).pipe(
        Effect.map(({ entered }) => entered),
        Effect.orElseSucceed(() => Option.none<DrafterCandidate>()),
      ),
    })
    return Option.match(Option.filter(entered, ({ reference }) => !listed.has(drafterReferenceKey(reference))), {
      onNone: () => candidates,
      onSome: (candidate) => [...candidates, candidate],
    })
  })

  const listDrafterCandidates = (target: DrafterTarget, other: Option.Option<string>) => Effect.gen(function* () {
    const listed = yield* listNativeDrafterCandidates(target, other)
    if (target._tag !== "Model" || Option.isSome(other)) return listed
    return { candidates: yield* withPinnedDrafter(target.modelId, target, listed.candidates), entered: Option.none() }
  })

  /** The drafter a choice resolves to; a rejected or unknown reference is refused, never attached. */
  const chooseDrafter = (target: DrafterTarget, choice: DrafterChoice) => Effect.gen(function* () {
    if (choice._tag === "None") return Option.none<DrafterCandidate>()
    const { candidates } = yield* listDrafterCandidates(target, Option.none())
    if (choice._tag === "Auto") return preselectedDrafterCandidate(candidates)
    const key = drafterReferenceKey(choice.reference)
    const matching = (list: readonly DrafterCandidate[]) =>
      Option.fromNullable(list.find(({ reference }) => drafterReferenceKey(reference) === key))
    const candidate = yield* Option.match(matching(candidates), {
      onSome: Effect.succeed,
      onNone: () => Option.match(otherReference(choice.reference), {
        onNone: () => Effect.fail(new InvalidModelReference({ reference: key, message: "This drafter isn't available for this model." })),
        onSome: (other) => listNativeDrafterCandidates(target, Option.some(other)).pipe(
          Effect.flatMap(({ entered }) => Option.match(entered, {
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

  /**
   * Fetches only the drafter its persisted choice resolves to, for a model whose target is already on this computer.
   * A separate drafter already on this computer, in any location, is used where it is; built-in heads need nothing.
   */
  const startDrafterOnly = (modelId: CustomFormModelId, chosen: DrafterCandidate) =>
    chosen.onThisComputer || chosen.reference._tag === "Embedded"
      ? Effect.succeed(false)
      : downloads.start(modelId, Option.some("Drafter")).pipe(Effect.map(Option.isSome))


  /**
   * An added model pinned to the commit its newest download of its files fetched, when that download completed at
   * another commit (an update): `revision` and the projector move to it, and a drafter chosen from the same repository
   * moves along when the update fetched it there. ICN then serves the new copy and removes the one it replaced. A
   * cancelled or failed update leaves the model on its pinned commit, still serving.
   */
  const pinnedUpdate = (
    config: CustomModelsConfig,
    modelId: CustomFormModelId,
    all: readonly Native.ModelDownload[],
  ): CustomModelsConfig => {
    const added = addedModel(config, modelId)
    const latest = all.flatMap((download) => sameModelId(download.forModel, modelId)
      ? Option.toArray(Option.map(addedTargetPackage(modelId, download), (target) => ({ download, target })))
      : []).at(-1)
    if (Option.isNone(added) || latest?.download.state._tag !== "Completed") return config
    const [key, setting] = added.value
    const { download, target } = latest
    if (target.source._tag !== "HuggingFace" || target.source.revision === setting.revision) return config
    const { repository, revision: commit } = target.source
    const pinned: AddedModelSetting = {
      revision: commit,
      projector: Option.fromNullable(target.files.find(({ role }) => role === "projector")?.path),
    }
    const drafter = configuredDrafter(config, modelId)
    const fetchedDraft = download.bundle._tag === "SpeculativeDecoding" && download.bundle.draftSource._tag === "Separate"
      ? Option.some(download.bundle.draftSource.draft)
      : Option.none()
    const moved = Option.flatMap(Option.zipWith(drafter, fetchedDraft, (setting, draft) => [setting, draft] as const),
      ([setting, draft]) => setting._tag === "HuggingFace" && setting.repository === repository
        && packageHoldsFile(draft, setting.repository, commit, setting.path)
        ? Option.some({ ...setting, revision: commit })
        : Option.none())
    return {
      ...config,
      added: { ...config.added, [key]: pinned },
      drafters: Option.match(moved, {
        onNone: () => config.drafters,
        onSome: (setting) => ({ ...config.drafters, [modelId]: setting }),
      }),
    }
  }

  /** Pins every added model whose update finished downloading. */
  const pinFinishedUpdates = Effect.gen(function* () {
    const all = (yield* downloads.get).state.downloads
    const changed = yield* customModels.modify((config) => {
      const next = (Object.keys(config.added) as CustomFormModelId[])
        .reduce((current, modelId) => pinnedUpdate(current, modelId, all), config)
      return Effect.succeed([next !== config, next] as const)
    })
    if (changed) yield* settle
  })

  /** Stores a model's drafter choice, and the model itself when it was just added; a model chosen is never hidden. */
  const persistDrafter = (
    modelId: CustomFormModelId,
    choice: DrafterChoice,
    added: Option.Option<readonly [HuggingFaceFormModelId, AddedModelSetting]>,
  ) =>
    customModels.modify((config) => Effect.succeed([undefined, {
      ...withDrafterChoice(config, modelId, choice),
      added: Option.match(added, {
        onNone: () => config.added,
        onSome: ([id, setting]) => ({ ...config.added, [id]: setting }),
      }),
      hidden: config.hidden.filter((hidden) => !sameModelId(hidden, modelId)),
    } satisfies CustomModelsConfig] as const))

  // A model-location command changes nothing but the configuration, so it pushes and fails when ICN refuses it.
  const settle = reconciler.push.pipe(Effect.catchAll((error) => Effect.logWarning(error.message)))

  /** After a model is removed, ICN deletes Magnitude's drafter packages no model uses. */
  const pruneDrafters = localModels_.pruneDrafters.pipe(
    Effect.asVoid,
    Effect.catchAll((error) => Effect.logWarning(`Removing unused drafters failed: ${error.message}`)),
  )

  const toMutation = (cause: GeneratedClientError<unknown> | CustomModelMutationError): Effect.Effect<never, CustomModelMutationError> =>
    cause._tag === "InvalidModelReference" || cause._tag === "ModelRepositoryNotFound" || cause._tag === "ModelSourceUnavailable"
      || cause._tag === "LocalModelMutationFailed" || cause._tag === "ModelVersionNotRunnable" || cause._tag === "DrafterNotUsable"
      || cause._tag === "HuggingFaceAccessRequired"
      ? Effect.fail(cause)
      : candidateFailed("model")(cause)

  const rowFor = (modelId: ModelId) => localModels.state.pipe(Effect.map((state) => Option.fromNullable(
    [...state.models, ...state.hidden].find((model): model is CustomLocalModel =>
      model._tag === "Custom" && sameModelId(model.modelId, modelId)),
  )))

  /** A catalog model's ID for one of its aliases: the ID its files have as a found model. */
  const catalogModelForAlias = (modelId: ModelId) => sources.state.pipe(Effect.map(({ catalogModels }) => Option.fromNullable(
    catalogModels.find(({ source }) => Option.exists(source.aliases, (aliases) => aliases.some((alias) => sameModelId(alias, modelId))))?.id)))

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

  /** A catalog model whose weights are all in an external location is a found copy: hidden, never deleted. */
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

  /** Ends a removed model's downloads: running ones are cancelled and failed ones dismissed, so none resumes. */
  const endDownloads = (related: readonly RelatedDownload[]) => Effect.forEach(related, ({ download }) =>
    downloadIsActive(download)
      ? downloads.cancel(download.id).pipe(Effect.ignore)
      : downloadHasUnacknowledgedFailure(download)
        ? downloads.acknowledgeFailure(download.id).pipe(Effect.ignore)
        : Effect.void,
  { discard: true })

  const commands = CustomModelCommands.of({
    searchHuggingFace: ({ query, limit }) => icn.search({
      query,
      limit: Option.getOrElse(limit, () => 20),
    }).pipe(
      Effect.catchAll(candidateFailed(query)),
      Effect.map(({ results }) => ({
        results: results.flatMap((result) => {
          const repository = Schema.decodeUnknownOption(HuggingFaceRepositoryIdSchema)(result.repository)
          if (Option.isNone(repository)) return []
          return Option.toArray(Schema.decodeUnknownOption(HuggingFaceSearchResultSchema)({
            repository: repository.value,
            displayName: repositoryDisplayName(repository.value),
            uploader: repository.value.split("/")[0],
            ...presentField("variantOf", result.variantOf),
            downloads: Option.getOrElse(result.downloads, () => 0),
            gated: result.gated,
            ...presentField("versionCount", result.versionCount),
            ...presentField("architecture", Option.filter(result.architecture, (value) => value.length > 0)),
            ...presentField("unsupported", Option.some(result.unsupported).pipe(Option.filter((reasons) => reasons.length > 0),
              Option.map((reasons) => productUnsupportedReasons(reasons, "")))),
          }))
        }),
      })),
    ),

    inspectCandidate: inspect,

    inspectCandidateWithProgress: inspectWithProgress,

    listDrafterCandidates,

    addHuggingFaceModel: ({ repository, revision, version, drafter, force }) => Effect.gen(function* () {
      const reference = Option.match(version, { onNone: () => repository, onSome: (quant) => `${repository}:${quant}` })
      const native = yield* icn.inspect({ _tag: "HuggingFace", repository, revision: Option.getOrElse(revision, () => "main") })
        .pipe(Effect.catchAll(candidateFailed(reference)))
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
      const added = Option.isSome(existing)
        ? Option.none()
        : Option.map(Schema.decodeUnknownOption(HuggingFaceFormModelIdSchema)(modelId), (id) => [id, {
            revision: source.commit,
            projector: Option.map(inspection.projector, ({ path }) => path),
          } satisfies AddedModelSetting] as const)
      // Auto stays Auto, so the model follows automatic choice (a catalog drafter by lineage included); only a
      // drafter the user names is pinned.
      yield* persistDrafter(modelId, persistedChoice(drafter, chosen, Option.none()), added)
      yield* settle
      // An added model downloads its files with the drafter its choice resolves to.
      const drafterDownloadStarted = Option.isSome(added)
        ? yield* downloads.start(modelId, Option.none()).pipe(Effect.as(false))
        : yield* Option.match(chosen, { onNone: () => Effect.succeed(false), onSome: (value) => startDrafterOnly(modelId, value) })
      return {
        _tag: "Custom" as const,
        modelId,
        version: selected.label,
        drafter: chosen,
        outcome: Option.isSome(existing) ? "AlreadyOnThisComputer" as const : "Started" as const,
        drafterDownloadStarted,
      }
    }).pipe(Effect.catchAll(toMutation)),

    setModelDrafter: (modelId, drafter) => Effect.gen(function* () {
      const row = yield* rowFor(modelId)
      if (Option.isNone(row)) return yield* notFound(modelId)
      const chosen = yield* chooseDrafter({ _tag: "Model", modelId }, drafter)
      const recommended = Option.map(row.value.recommendedDrafter, ({ reference }) => reference)
      yield* customModels.modify((config) => Effect.succeed([undefined,
        withDrafterChoice(config, modelId, persistedChoice(drafter, chosen, recommended))] as const))
      // Changing or turning off a drafter only changes the choice: the drafter stays on disk, so turning it back on
      // downloads nothing. Unused drafters go when a model is removed.
      yield* settle
      const drafterDownloadStarted = yield* Option.match(chosen, {
        onNone: () => Effect.succeed(false),
        onSome: (value) => startDrafterOnly(modelId, value),
      })
      return { drafter: chosen, drafterDownloadStarted }
    }).pipe(Effect.catchAll(toMutation)),

    hide: (requested) => Effect.gen(function* () {
      // An alias names its catalog model. A catalog model found only in an external location is hidden by its catalog
      // ID, and ICN then stops attributing its copies to it; hiding it again changes nothing.
      const modelId: ModelId = Option.getOrElse(yield* catalogModelForAlias(requested), () => requested)
      const config = yield* customModels.get
      if (Schema.is(CatalogFormModelIdSchema)(modelId)) {
        if (!(yield* isKnownCatalogModel(modelId))) return yield* notFound(modelId)
        if (!isHidden(config, modelId) && Option.isNone(yield* externalCatalogFiles(modelId))) {
          return yield* catalogHideRefusal(modelId)
        }
      } else {
        if (Option.isSome(addedModel(config, modelId))) return yield* managedHideRefusal(modelId)
        if (Option.isNone(yield* rowFor(modelId))) return yield* notFound(modelId)
      }
      yield* customModels.modify((current) => Effect.succeed([undefined, isHidden(current, modelId)
        ? current
        : { ...current, hidden: [...current.hidden, modelId] }] as const))
      yield* settle
    }),

    externalCatalogFiles,

    unhide: (requested) => Effect.gen(function* () {
      // An alias names its catalog model, which shows again with every copy hidden by the ID it has as a found model.
      const modelId: ModelId = Option.getOrElse(yield* catalogModelForAlias(requested), () => requested)
      const shownIds: readonly ModelId[] = Schema.is(CatalogFormModelIdSchema)(modelId)
        ? [modelId, ...yield* catalogAliases(modelId)]
        : [modelId]
      yield* customModels.modify((config) => Effect.succeed([undefined, {
        ...config,
        hidden: config.hidden.filter((hidden) => !shownIds.some((shown) => sameModelId(shown, hidden))),
      }] as const))
      yield* settle
    }),

    addFolder: (path) => validateFolder(path).pipe(
      Effect.flatMap((folder) => customModels.modify((config) => Effect.succeed([undefined, config.paths.includes(folder)
        ? config
        : { ...config, paths: [...config.paths, folder] }] as const))),
      Effect.zipRight(reconciler.push),
    ),

    removeFolder: (path) => Effect.gen(function* () {
      const trimmed = path.trim()
      if (!isAbsolute(trimmed)) return yield* new InvalidDirectoryPath({ path })
      const folder = normalize(trimmed)
      yield* customModels.modify((config) => Effect.succeed([undefined, {
        ...config,
        paths: config.paths.filter((existing) => existing !== folder && existing !== trimmed),
      }] as const))
      yield* reconciler.push
    }),

    setLocationEnabled: (kind, enabled) => customModels.modify((config) => Effect.succeed([undefined,
      withLocationEnabled(config, kind, enabled)] as const)).pipe(Effect.zipRight(reconciler.push)),

    listLocations: customModels.get.pipe(Effect.flatMap(locations.list)),

    remove: (modelId) => Effect.gen(function* () {
      const { config, related } = yield* modelAcquisition(modelId)
      const added = addedModel(config, modelId)
      if (Option.isNone(added)) {
        if (Option.isNone(yield* rowFor(modelId))) return yield* notFound(modelId)
        yield* endDownloads(related)
        yield* customModels.modify((current) => Effect.succeed([undefined, isHidden(current, modelId)
          ? current
          : { ...current, hidden: [...current.hidden, modelId] }] as const))
        yield* settle
        return {}
      }
      yield* endDownloads(related)
      // ICN deletes only Magnitude's copy; the same bytes elsewhere remain and the model stays as found.
      yield* localModels_.removeFiles(modelId).pipe(
        Effect.catchIf((error) => error._tag === "GeneratedClientRemoteError" && error.status === 404, () => Effect.void),
        Effect.mapError(commandFailed("remove")),
      )
      yield* customModels.modify((current) => Effect.succeed([undefined, {
        ...withDrafterChoice(current, modelId, { _tag: "Auto" }),
        added: withoutKey(current.added, added.value[0]),
      }] as const))
      yield* settle
      yield* pruneDrafters
      return {}
    }),

    sync: (modelId) => Effect.gen(function* () {
      const row = yield* rowFor(modelId)
      const { config, related } = yield* modelAcquisition(modelId)
      const added = Option.isSome(addedModel(config, modelId))
      if (Option.isNone(row) && !added) return yield* notFound(modelId)
      const attached = Option.flatMap(row, ({ drafter }) => drafter)
      const acquisition = Option.map(row, ({ acquisitionState }) => acquisitionState._tag)
      const targetActive = Option.exists(currentDownload(related, "Target"), downloadIsActive)
      const targetMissing = added && !targetActive
        && Option.match(acquisition, { onNone: () => true, onSome: (tag) => tag === "NotInstalled" || tag === "InstallFailed" })
      const current = { outcome: "AlreadyCurrent" as LocalModelSyncOutcome }
      if (targetMissing) {
        return Option.isSome(yield* downloads.start(modelId, Option.none())) ? { outcome: "Installing" as const } : current
      }
      const updateCommit = Option.flatMap(Option.fromNullable((yield* sources.state).customModels.find(({ id }) => id === modelId)),
        ({ source }) => source.updateCommit)
      if (added && Option.isSome(updateCommit) && !targetActive
        && Option.exists(acquisition, (tag) => tag === "UpdateAvailable" || tag === "UpdateFailed")) {
        // ICN fetches the update at the commit its repository has moved to; the model keeps serving until it lands.
        yield* downloads.start(modelId, Option.some("Update"))
        return { outcome: "Updating" as const }
      }
      const drafterUpdate = Option.exists(attached, ({ updateAvailable }) => updateAvailable)
      const drafterMissing = Option.exists(attached, ({ availability }) => availability._tag === "DownloadFailed")
      const drafterActive = Option.exists(currentDownload(related, "Drafter"), downloadIsActive)
      if ((!drafterUpdate && !drafterMissing) || drafterActive || targetActive) return current
      const started = Option.isSome(yield* downloads.start(modelId, Option.some("Drafter")))
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

  // ICN resumes every requested download a restart interrupted, an update included, and drops a cancelled one, so
  // ACN starts none itself; each update is pinned as its download finishes.
  yield* downloads.changes.pipe(
    Stream.runForEach(() => pinFinishedUpdates.pipe(Effect.catchAll((error) => Effect.logWarning(
      `Unable to pin a finished update: ${error.message}`,
    )))),
    Effect.forkScoped,
  )
  // Stored IDs follow the models they name: a catalog model's alias becomes its catalog ID once the catalog lists
  // it, and an older `local:` label the current one.
  yield* localModels.changes.pipe(
    Stream.runForEach(() => Effect.gen(function* () {
      const { models, hidden } = yield* localModels.state
      yield* customModels.modify((config) => Effect.succeed([undefined,
        withCanonicalReferences(config, (modelId) => canonicalLocalModelId([...models, ...hidden], modelId))] as const))
    }).pipe(
      Effect.catchAll((error) => Effect.logWarning(`Unable to rename stored model references: ${error.message}`)),
    )),
    Effect.forkScoped,
  )
  return commands
}))
