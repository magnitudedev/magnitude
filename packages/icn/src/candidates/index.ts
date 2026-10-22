import { Context, Effect, Layer, Stream } from "effect"
import type {
  CandidateInspection,
  CandidateInspectionEvent,
  CandidateInspectionRequest,
  DrafterCandidatesRequest,
  DrafterCandidatesResponse,
  HuggingFaceModelSearchRequest,
  HuggingFaceModelSearchResults,
  HuggingFaceRepositorySnapshot,
  ModelAssessment,
  ModelCandidate,
} from "@magnitudedev/icn-protocol/schemas"
import { IcnClient, type IcnClientService } from "../client.js"

type Failure<K extends keyof IcnClientService, O extends keyof IcnClientService[K]> =
  IcnClientService[K][O] extends (...args: never[]) => Effect.Effect<unknown, infer E, unknown> ? E : never

export type HuggingFaceSearchError = Failure<"huggingFace", "searchHuggingFaceModels">
export type HuggingFaceResolveError = Failure<"huggingFace", "resolveHuggingFaceRepository">
export type CandidateInspectionError = Failure<"candidates", "inspectModelCandidate">
export type CandidateInspectionStreamError = Effect.Effect.Error<
  ReturnType<IcnClientService["candidates"]["streamModelCandidateInspection"]>
>
export type DrafterCandidatesError = Failure<"candidates", "listDrafterCandidates">
export type CandidateAssessmentError = Failure<"models", "assessModelCandidate">

/**
 * Files not yet added: Hugging Face search, repository resolution (a revision pinned to its commit),
 * inspection (alone, or as its progress then itself), drafters in ICN's category order (with the listed one a
 * different drafter resolves to), and assessment.
 */
export interface IcnCandidatesService {
  readonly search: (request: HuggingFaceModelSearchRequest) => Effect.Effect<HuggingFaceModelSearchResults, HuggingFaceSearchError>
  readonly resolveRepository: (
    repository: string,
    revision: string,
  ) => Effect.Effect<HuggingFaceRepositorySnapshot, HuggingFaceResolveError>
  readonly inspect: (request: CandidateInspectionRequest) => Effect.Effect<CandidateInspection, CandidateInspectionError>
  readonly inspectWithProgress: (
    request: CandidateInspectionRequest,
  ) => Stream.Stream<CandidateInspectionEvent, CandidateInspectionStreamError>
  readonly drafters: (request: DrafterCandidatesRequest) => Effect.Effect<DrafterCandidatesResponse, DrafterCandidatesError>
  readonly assess: (candidate: ModelCandidate) => Effect.Effect<ModelAssessment, CandidateAssessmentError>
}
export class IcnCandidates extends Context.Tag("@magnitudedev/icn/IcnCandidates")<IcnCandidates, IcnCandidatesService>() {}

export const IcnCandidatesLive: Layer.Layer<IcnCandidates, never, IcnClient> = Layer.effect(
  IcnCandidates,
  Effect.map(IcnClient, (client) => IcnCandidates.of({
    search: (request) => client.huggingFace.searchHuggingFaceModels({ payload: request }),
    resolveRepository: (repository, revision) =>
      client.huggingFace.resolveHuggingFaceRepository({ payload: { repository, revision } }),
    inspect: (request) => client.candidates.inspectModelCandidate({ payload: request }),
    inspectWithProgress: (request) => Stream.unwrap(Effect.map(
      client.candidates.streamModelCandidateInspection({ payload: request }),
      (response) => response.events,
    )),
    drafters: (request) => client.candidates.listDrafterCandidates({ payload: request }),
    assess: (candidate) => client.models.assessModelCandidate({ payload: candidate }),
  })),
)
