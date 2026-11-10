import { Option } from "effect"
import type { CustomLocalModel, LocalModel, LocalModelAssessment, LocalModelDeveloper, LocalModelFamily, RecommendedModel, SpeculativeMethod, UnsupportedLocalModel } from "@magnitudedev/sdk"
import { labForDeveloper } from "../model-logo"
import { quantBits } from "./parts"
import type { DiscoveredModel } from "./onboarding"
import type { OfferedModel } from "./expanding-model"
import type { RunnableFamily } from "./explore-modal"
import type { OwnedModel, UnrunnableFile } from "./your-models"
import type { PlacementPhrase, PlacementTier, QuantView } from "./view"

const methodName: Record<SpeculativeMethod["_tag"], string> = { Mtp: "MTP", DFlash: "DFlash", DFlash2: "DFlash 2", DSpark: "DSpark", LiLiCorr: "LiLiCorr" }

const foundSourceName = { HuggingFaceCache: "HF cache", LmStudio: "LM Studio", LlamaCpp: "llama.cpp", Ollama: "Ollama", Folder: "Folder" } as const

const developerOf = (model: LocalModel): Option.Option<LocalModelDeveloper> =>
  model._tag === "Catalog" ? Option.some(model.catalogData.developer) : model.presentation.developer

/** The Hugging Face repository a model comes from: an `hf:owner/repo/…` id, an added repository, or a catalog source URL. */
const repositoryOf = (model: LocalModel): string | null => {
  if (model._tag === "Custom") {
    if (model.origin._tag === "Added") return model.origin.repository
    const [form, owner, repo] = model.modelId.split(/[:/]/)
    return form === "hf" && owner && repo ? `${owner}/${repo}` : null
  }
  return model.catalogData.quant.repository
}

const quantNameOf = (model: LocalModel): string => model._tag === "Catalog" ? model.catalogData.quant.name : model.presentation.variantLabel

const assessmentOf = (model: LocalModel): LocalModelAssessment | null =>
  model.servingState._tag === "Assessed" ? model.servingState.assessment : null

export const phraseOf = (assessment: Extract<LocalModelAssessment, { _tag: "Fits" }>, unified: boolean): PlacementPhrase =>
  assessment.residency._tag === "Resident"
    ? unified ? "Fits in memory" : "Fits in GPU"
    : assessment.placement.diskBytes > 0 && assessment.residency.bank.coldExperts === "disk" ? "Streams from SSD" : "Streams from RAM"

const tiersOf = (assessment: Extract<LocalModelAssessment, { _tag: "Fits" }>, unified: boolean): readonly PlacementTier[] => [
  { kind: unified ? "Unified" : "Device", bytes: assessment.placement.deviceBytes },
  { kind: "Host", bytes: assessment.placement.hostBytes },
  { kind: "Disk", bytes: assessment.placement.diskBytes },
]

/** The estimate at the context the speed figure stands for: the sample nearest min(50K, served). */
const speedOf = (assessment: Extract<LocalModelAssessment, { _tag: "Fits" }>): number | null => {
  const performance = assessment.residency.performance
  if (performance._tag !== "Estimated" || performance.samples.length === 0) return null
  const target = Math.min(50_000, assessment.served.contextLength)
  return performance.samples.reduce((best, sample) => Math.abs(sample.contextTokens - target) < Math.abs(best.contextTokens - target) ? sample : best).estimatedTokensPerSecond
}

/** One quant as this machine would run it, from its assessment. */
export const quantView = (quant: { readonly key: string; readonly name: string; readonly repository: string | null; readonly sizeBytes: number; readonly downloadBytes: number }, assessment: LocalModelAssessment | null, unified: boolean, measuring = false): QuantView => {
  const fits = assessment?._tag === "Fits" ? assessment : null
  return {
    key: quant.key,
    bits: quantBits(quant.name),
    publisher: quant.repository?.split("/")[0] ?? null,
    name: quant.name,
    repository: quant.repository,
    sizeBytes: quant.sizeBytes,
    downloadBytes: quant.downloadBytes,
    fits: fits !== null,
    measuring: measuring || (fits !== null && fits.residency.performance._tag === "Pending"),
    speed: fits ? speedOf(fits) : null,
    phrase: fits ? phraseOf(fits, unified) : null,
    placement: fits ? tiersOf(fits, unified) : [],
    servedContext: fits?.served.contextLength ?? null,
    quality: null,
  }
}

/** A model on this computer as a quant this machine runs. */
export const quantOf = (model: LocalModel, unified: boolean): QuantView =>
  quantView({ key: model.modelId, name: quantNameOf(model), repository: repositoryOf(model), sizeBytes: model.storageBytes, downloadBytes: model.storageBytes }, assessmentOf(model), unified, model.servingState._tag === "Assessing")

const directoryOf = (path: string) => path.replace(/[\\/][^\\/]*$/, "")

const locationOf = (model: LocalModel): string | null => {
  const state = model.acquisitionState
  return "installation" in state && state.installation._tag === "Resolved" ? directoryOf(state.installation.primaryPath) : null
}

const drafterOf = (model: LocalModel): OwnedModel["drafter"] => {
  if (model._tag === "Custom") {
    const attached = Option.getOrNull(model.drafter)
    if (attached) return { method: attached.label, sizeBytes: attached.sizeBytes, attached: true }
    const recommended = Option.getOrNull(model.recommendedDrafter)
    return recommended ? { method: Option.match(recommended.method, { onNone: () => recommended.label, onSome: method => methodName[method._tag] }), sizeBytes: recommended.sizeBytes, attached: false } : null
  }
  if (model.drafterWithheld) return null
  return Option.match(model.catalogData.drafter, { onNone: () => null, onSome: drafter => ({ method: methodName[drafter.method._tag], sizeBytes: drafter.sizeBytes, attached: true }) })
}

/** A model on this computer as the Your models list shows it. */
export const ownedModel = (model: LocalModel, unified: boolean): OwnedModel => {
  const developer = Option.getOrNull(developerOf(model))
  const assessment = assessmentOf(model)
  const quant = quantOf(model, unified)
  return {
    modelId: model.modelId,
    name: model.presentation.displayName,
    lab: developer ? labForDeveloper(developer.lab) : undefined,
    country: developer?.country ?? null,
    developer: developer?.name ?? null,
    license: Option.getOrNull(model.presentation.license),
    quant,
    contextLimit: assessment?._tag === "Fits" ? assessment.profile.contextLength : null,
    location: locationOf(model),
    drafter: drafterOf(model),
    searchText: `${model.presentation.displayName} ${quant.repository ?? ""} ${quant.name}`,
  }
}

/** A model found in another app's folder, as onboarding lists it. */
export const discoveredModel = (model: CustomLocalModel): DiscoveredModel | null => {
  if (model.origin._tag !== "Found") return null
  const developer = Option.getOrNull(model.presentation.developer)
  return {
    key: model.modelId,
    name: model.presentation.displayName,
    lab: developer ? labForDeveloper(developer.lab) : undefined,
    bits: quantBits(model.presentation.variantLabel),
    source: model.origin.source._tag === "Folder" ? directoryOf(model.origin.source.path).split(/[\\/]/).pop() ?? "Folder" : foundSourceName[model.origin.source._tag],
    fromHuggingFaceCache: model.origin.source._tag === "HuggingFaceCache",
    warning: model.modelId.startsWith("local:")
      ? Option.isSome(model.presentation.variantOf) ? "Source unknown: Magnitude knows which model this is but not which Hugging Face repo it came from." : "Unknown model: Magnitude can’t tell which model this file is."
      : null,
  }
}

/** A file Magnitude found but can't run, and why. */
export const unrunnableFile = (model: UnsupportedLocalModel): UnrunnableFile => {
  const architecture = model.reasons.find(reason => reason._tag === "Architecture")
  return {
    name: model.displayName,
    reason: architecture && "architecture" in architecture ? `${architecture.architecture} architecture` : model.detail,
  }
}

/** A recommended tested model as an offered model: its ladder's rungs as this machine would run them, the chosen one marked. */
export const offeredFromRecommendation = (recommendation: RecommendedModel, models: ReadonlyMap<string, LocalModel>, unified: boolean): OfferedModel | null => {
  const rungs = recommendation.rungs.flatMap(id => { const model = models.get(id); return model ? [model] : [] })
  const first = rungs[0]
  if (!first || first._tag !== "Catalog") return null
  const drafter = Option.getOrNull(first.catalogData.drafter)
  const assessment = rungs.map(assessmentOf).find(entry => entry?._tag === "Fits")
  return {
    key: recommendation.base,
    catalog: true,
    name: recommendation.name,
    lab: labForDeveloper(recommendation.developer.lab),
    country: recommendation.developer.country,
    fineTuneOf: null,
    drafter: drafter ? { method: methodName[drafter.method._tag], sizeBytes: drafter.sizeBytes } : null,
    contextLimit: assessment?._tag === "Fits" ? assessment.profile.contextLength : null,
    quants: rungs.map(model => ({ ...quantOf(model, unified), downloadBytes: model.storageBytes + (drafter?.sizeBytes ?? 0) })),
    recommended: Option.getOrNull(recommendation.chosen),
  }
}

/** A family Magnitude runs, for the "What runs in Magnitude" list. */
export const runnableFamily = (family: LocalModelFamily): RunnableFamily => ({
  lab: labForDeveloper(family.developer.lab),
  models: family.name,
  comingSoon: family.status === "ComingSoon",
})
