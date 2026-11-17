import { useState } from "react"
import type { ServingMemory } from "@magnitudedev/client-common"
import { cardClass } from "./parts"
import { OfferedModelsSkeleton } from "./skeletons"
import { useDownload } from "./explore"
import { RecommendedForYourMachine, useRecommendedSelection } from "./recommended-for-your-machine"
import { recommendedForYourMachine, useRecommendations } from "./recommendations"

const NO_DRAFTER_CHOICES = {}
const NOTHING_OWNED: ReadonlySet<string> = new Set()

/**
 * The Models page before any model is downloaded: it says so, then offers what onboarding recommends for this machine,
 * each opened model with its Download. A download that starts lists its model, which then takes this place.
 */
export function NoModelsYet({ memory }: { memory: ServingMemory }) {
  const [usOnly, setUsOnly] = useState(false)
  const view = useRecommendations(usOnly, memory)
  const recommendations = recommendedForYourMachine(view.offered, NOTHING_OWNED)
  const selection = useRecommendedSelection(recommendations)
  const download = useDownload(NO_DRAFTER_CHOICES)
  const placeholder = !view.ready || (view.measuring && recommendations.length === 0) ? <OfferedModelsSkeleton firstOpened footer /> : null
  return <>
    <p className={`${cardClass} px-4 py-3.5 text-[13px] text-slate-700 dark:text-slate-300`}>No models are downloaded yet. Download one to start chatting.</p>
    <RecommendedForYourMachine recommendations={recommendations} placeholder={placeholder} measuring={view.measuring}
      usOnly={usOnly} setUsOnly={setUsOnly} owned={NOTHING_OWNED} selection={selection} footer={download} />
  </>
}
