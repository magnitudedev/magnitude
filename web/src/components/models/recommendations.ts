import { useMemo } from "react"
import { Option } from "effect"
import { Result, useAtomValue } from "@effect-atom/atom-react"
import { CountryCodeSchema, type LocalModel, type RecommendedModel } from "@magnitudedev/sdk"
import { useAgentClient, useLocalModels, type ServingMemory } from "@magnitudedev/client-common"
import { offeredFromRecommendation, runnableFamily } from "./data"
import type { OfferedModel } from "./expanding-model"
import type { RunnableFamily } from "./explore-modal"

export interface OfferedRecommendation {
  readonly recommendation: RecommendedModel
  readonly model: OfferedModel
}

export interface RecommendationsView {
  readonly ready: boolean
  readonly measuring: { readonly settled: number; readonly total: number } | null
  readonly offered: readonly OfferedRecommendation[]
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
    if (!Result.isSuccess(recommendations) || !Result.isSuccess(models)) return { ready: false, measuring: null, offered: [], families: [] }
    const { complete, measured, rungs } = recommendations.value
    const byId = new Map<string, LocalModel>(models.value.models.map(model => [model.modelId, model]))
    return {
      ready: true,
      measuring: complete ? null : { settled: measured, total: rungs },
      offered: recommendations.value.models.flatMap(recommendation => {
        const model = offeredFromRecommendation(recommendation, byId, memory)
        return model ? [{ recommendation, model }] : []
      }),
      families: recommendations.value.families.map(runnableFamily),
    }
  }, [recommendations, models, memory])
}
