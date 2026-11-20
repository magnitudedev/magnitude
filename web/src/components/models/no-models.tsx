import { useState } from "react"
import type { ServingMemory } from "@magnitudedev/client-common"
import { OfferedModelsSkeleton } from "./skeletons"
import { useDownload } from "./explore"
import { RecommendedForYourMachine, useRecommendedSelection } from "./recommended-for-your-machine"
import { recommendationsSettled, recommendedForYourMachine, useRecommendations } from "./recommendations"

const NO_DRAFTER_CHOICES = {}
const NOTHING_OWNED: ReadonlySet<string> = new Set()

/**
 * The Models page before any model is downloaded: a quiet note below the heading, then what onboarding recommends for
 * this machine, each opened model with its Download. A download that starts lists its model, which then takes this
 * place.
 */
export function NoModelsYet({ memory }: { memory: ServingMemory }) {
  const [usOnly, setUsOnly] = useState(false)
  const view = useRecommendations(usOnly, memory)
  const recommendations = recommendedForYourMachine(view.offered, NOTHING_OWNED)
  const selection = useRecommendedSelection(recommendations)
  const download = useDownload(NO_DRAFTER_CHOICES)
  // The list shows once every quant has been measured, in one go; until then, its skeleton at the list's own size.
  const placeholder = recommendationsSettled(view) ? null : <OfferedModelsSkeleton firstOpened footer />
  return <>
    <p className="text-sm text-slate-500 dark:text-slate-400">No models yet. Download one below, or explore more on Hugging Face.</p>
    <RecommendedForYourMachine recommendations={recommendations} placeholder={placeholder} measuring={view.measuring}
      usOnly={usOnly} setUsOnly={setUsOnly} owned={NOTHING_OWNED} selection={selection} footer={download} className="mt-6" />
  </>
}
