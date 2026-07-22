export {
  ModelRecipeFitClass as LocalInferenceFitClass,
  ModelRecipeQuantization as LocalInferenceQuantization,
  ModelRecipeRecommendation as LocalModelRecommendation,
} from "@magnitudedev/icn/recipes"

export type {
  HardwareSnapshotSchema as IcnHardwareState,
  ModelList as IcnInventoryState,
} from "@magnitudedev/icn/generated"
export type { ModelRecipesState } from "@magnitudedev/icn/recipes"
