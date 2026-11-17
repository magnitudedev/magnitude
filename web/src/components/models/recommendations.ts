import { useMemo } from "react"
import { Option } from "effect"
import { Result, useAtomValue } from "@effect-atom/atom-react"
import { CountryCodeSchema, type LocalModel, type RecommendedModel, type RoomShortfall } from "@magnitudedev/sdk"
import { useAgentClient, useLocalModels, type ServingMemory } from "@magnitudedev/client-common"
import { offeredFromRecommendation, runnableFamily } from "./data"
import type { OfferedModel } from "./expanding-model"
import type { RunnableFamily } from "./explore-view"

export interface OfferedRecommendation {
  readonly recommendation: RecommendedModel
  readonly model: OfferedModel
}

/** One offered model with what is said beside its name. */
export interface Recommendation {
  readonly model: OfferedModel
  readonly note: string | undefined
}

/**
 * What "Recommended for your machine" offers, on onboarding and on an empty Models page alike: up to three models fast
 * enough to start with, each with its note. A model with a quant already here says so; otherwise the first is the
 * smartest that runs well and the fastest is named.
 */
export const recommendedForYourMachine = (offered: readonly OfferedRecommendation[], owned: ReadonlySet<string>): readonly Recommendation[] => {
  const choices = offered.filter(entry => entry.recommendation.meetsOnboardingFloor).slice(0, 3)
  const speed = (entry: OfferedRecommendation) => entry.model.quants.find(quant => quant.key === entry.model.recommended)?.speed ?? 0
  const fastest = choices.reduce<OfferedRecommendation | undefined>((best, entry) => best && speed(best) >= speed(entry) ? best : entry, undefined)
  return choices.map((entry, index) => {
    const have = entry.model.quants.find(quant => owned.has(quant.key))
    return { model: entry.model, note: have ? `You have the ${have.bits}-bit quant` : index === 0 ? "Smartest that runs well here" : entry === fastest ? "Fastest" : undefined }
  })
}

export interface RecommendationsView {
  readonly ready: boolean
  readonly measuring: { readonly settled: number; readonly total: number } | null
  readonly offered: readonly OfferedRecommendation[]
  readonly roomShortfall: RoomShortfall | null
  readonly families: readonly RunnableFamily[]
}

/**
 * The service's recommendations for this machine as offered models, optionally only US developers', and the families
 * it runs. While any rung is still being measured the order is not final, so `measuring` says how far along it is.
 */
export function useRecommendations(usOnly: boolean, memory: ServingMemory): RecommendationsView {
  const client = useAgentClient()
  const developerCountry = usOnly ? Option.some(CountryCodeSchema.make("US")) : Option.none()
  const recommendations = useAtomValue(client.Models.GetRecommendations({ developerCountry })).result
  const models = useLocalModels()
  return useMemo(() => {
    if (!Result.isSuccess(recommendations) || !Result.isSuccess(models)) return { ready: false, measuring: null, offered: [], roomShortfall: null, families: [] }
    const { complete, measured, rungs } = recommendations.value
    const byId = new Map<string, LocalModel>(models.value.models.map(model => [model.modelId, model]))
    return {
      ready: true,
      measuring: complete ? null : { settled: measured, total: rungs },
      offered: recommendations.value.models.flatMap(recommendation => {
        const model = offeredFromRecommendation(recommendation, byId, memory)
        return model ? [{ recommendation, model }] : []
      }),
      roomShortfall: Option.getOrNull(recommendations.value.roomShortfall),
      families: recommendations.value.families.map(runnableFamily),
    }
  }, [recommendations, models, memory])
}
