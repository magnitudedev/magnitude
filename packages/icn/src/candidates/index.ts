import { Context, Effect, Layer } from "effect"
import type {
  CandidateInspection,
  CandidateInspectionRequest,
  DrafterCandidatesRequest,
  DrafterCandidatesResponse,
  HuggingFaceModelSearchRequest,
  HuggingFaceModelSearchResults,
  HuggingFaceRepositorySnapshot,
  ModelAssessment,
  ModelCandidate,
  ServableModelBundle,
} from "@magnitudedev/icn-protocol/schemas"
import { IcnClient, type IcnClientService } from "../client.js"

type Failure<K extends keyof IcnClientService, O extends keyof IcnClientService[K]> =
  IcnClientService[K][O] extends (...args: never[]) => Effect.Effect<unknown, infer E, unknown> ? E : never

export type HuggingFaceSearchError = Failure<"huggingFace", "searchHuggingFaceModels">
export type HuggingFaceResolveError = Failure<"huggingFace", "resolveHuggingFaceRepository">
export type CandidateInspectionError = Failure<"candidates", "inspectModelCandidate">
export type DrafterCandidatesError = Failure<"candidates", "listDrafterCandidates">
export type CandidateAssessmentError = Failure<"models", "assessModelCandidate">
export type CandidateBundleError = Failure<"candidates", "buildCandidateBundle">

/**
 * Files not yet added: Hugging Face search, repository resolution (a revision pinned to its commit),
 * inspection, drafters in ICN's category order (with the listed one a different drafter resolves to), assessment, and the exact bundle a download fetches (a
 * local target yields its drafter alone).
 */
export interface IcnCandidatesService {
  readonly search: (request: HuggingFaceModelSearchRequest) => Effect.Effect<HuggingFaceModelSearchResults, HuggingFaceSearchError>
  readonly resolveRepository: (
    repository: string,
    revision: string,
  ) => Effect.Effect<HuggingFaceRepositorySnapshot, HuggingFaceResolveError>
  readonly inspect: (request: CandidateInspectionRequest) => Effect.Effect<CandidateInspection, CandidateInspectionError>
  readonly drafters: (request: DrafterCandidatesRequest) => Effect.Effect<DrafterCandidatesResponse, DrafterCandidatesError>
  readonly assess: (candidate: ModelCandidate) => Effect.Effect<ModelAssessment, CandidateAssessmentError>
  readonly bundle: (candidate: ModelCandidate) => Effect.Effect<ServableModelBundle, CandidateBundleError>
}
export class IcnCandidates extends Context.Tag("@magnitudedev/icn/IcnCandidates")<IcnCandidates, IcnCandidatesService>() {}

export const IcnCandidatesLive: Layer.Layer<IcnCandidates, never, IcnClient> = Layer.effect(
  IcnCandidates,
  Effect.map(IcnClient, (client) => IcnCandidates.of({
    search: (request) => client.huggingFace.searchHuggingFaceModels({ payload: request }),
    resolveRepository: (repository, revision) =>
      client.huggingFace.resolveHuggingFaceRepository({ payload: { repository, revision } }),
    inspect: (request) => client.candidates.inspectModelCandidate({ payload: request }),
    drafters: (request) => client.candidates.listDrafterCandidates({ payload: request }),
    assess: (candidate) => client.models.assessModelCandidate({ payload: candidate }),
    bundle: (candidate) => client.candidates.buildCandidateBundle({ payload: candidate }),
  })),
)
