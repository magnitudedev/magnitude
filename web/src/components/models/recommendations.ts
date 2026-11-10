import { useMemo } from "react"
import { Option } from "effect"
import { Result, useAtomValue } from "@effect-atom/atom-react"
import { CountryCodeSchema, type LocalModel, type RecommendedModel } from "@magnitudedev/sdk"
import { useAgentClient, useLocalModels } from "@magnitudedev/client-common"
import { offeredFromRecommendation, runnableFamily } from "./data"
import type { OfferedModel } from "./expanding-model"
import type { RunnableFamily } from "./explore-modal"

export interface OfferedRecommendation {
  readonly recommendation: RecommendedModel
  readonly model: OfferedModel
}

/** The service's recommendations for this machine as offered models, optionally only US developers', and the families it runs. */
export function useRecommendations(usOnly: boolean, unified: boolean): { readonly ready: boolean; readonly offered: readonly OfferedRecommendation[]; readonly families: readonly RunnableFamily[] } {
  const client = useAgentClient()
  const developerCountry = usOnly ? Option.some(CountryCodeSchema.make("US")) : Option.none()
  const recommendations = useAtomValue(client.Models.GetRecommendations({ developerCountry })).result
  const models = useLocalModels()
  return useMemo(() => {
    if (!Result.isSuccess(recommendations) || !Result.isSuccess(models)) return { ready: false, offered: [], families: [] }
    const byId = new Map<string, LocalModel>(models.value.models.map(model => [model.modelId, model]))
    return {
      ready: true,
      offered: recommendations.value.models.flatMap(recommendation => {
        const model = offeredFromRecommendation(recommendation, byId, unified)
        return model ? [{ recommendation, model }] : []
      }),
      families: recommendations.value.families.map(runnableFamily),
    }
  }, [recommendations, models, unified])
}
