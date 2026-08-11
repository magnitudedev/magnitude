import { Schema } from "effect"
import {
  LocalModelsStateSchema,
  type LocalModel,
  type LocalModelsState,
} from "@magnitudedev/sdk"

/** The model-menu surfaces consume the canonical state, not a copied DTO. */
export const selectModelMenusLocalModelsState = (
  state: LocalModelsState,
): LocalModelsState => state

const alignUndisplayedHeadroom = (
  previousModel: LocalModel | undefined,
  nextModel: LocalModel,
): LocalModel => {
  if (previousModel?.servingState._tag !== "Assessed"
    || previousModel.servingState.assessment._tag !== "Fits"
    || nextModel.servingState._tag !== "Assessed"
    || nextModel.servingState.assessment._tag !== "Fits") return nextModel
  const previousHeadroomState = previousModel.servingState.assessment.memory.currentHeadroomState
  const nextHeadroomState = nextModel.servingState.assessment.memory.currentHeadroomState
  if (previousHeadroomState._tag !== nextHeadroomState._tag
    || previousHeadroomState._tag === "NotObserved"
    || nextHeadroomState._tag === "NotObserved") return nextModel
  let alignedCurrentHeadroomState = nextHeadroomState
  if (nextHeadroomState._tag === "Sufficient"
    && previousHeadroomState._tag === "Sufficient") {
    alignedCurrentHeadroomState = {
      ...nextHeadroomState,
      observation: {
        ...nextHeadroomState.observation,
        allocationHeadroomBytes:
          previousHeadroomState.observation.allocationHeadroomBytes,
      },
    }
  } else if (nextHeadroomState._tag === "Insufficient"
    && previousHeadroomState._tag === "Insufficient") {
    alignedCurrentHeadroomState = {
      ...nextHeadroomState,
      observation: {
        ...nextHeadroomState.observation,
        allocationHeadroomBytes:
          previousHeadroomState.observation.allocationHeadroomBytes,
      },
      minimumAdditionalAvailableBytes:
        previousHeadroomState.minimumAdditionalAvailableBytes,
    }
  }
  return {
    ...nextModel,
    servingState: {
      ...nextModel.servingState,
      assessment: {
        ...nextModel.servingState.assessment,
        memory: {
          ...nextModel.servingState.assessment.memory,
          currentHeadroomState: alignedCurrentHeadroomState,
        },
      },
    },
  }
}

const localModelsStateEquivalent = Schema.equivalence(LocalModelsStateSchema)

/**
 * Models and Catalog display the headroom category, not each sampled byte
 * value. Byte-only polling updates are therefore equivalent for these two
 * surfaces; category, model, progress, assessment, and acquisition changes
 * remain observable.
 */
export const modelMenusLocalModelsStateEquivalent = (
  left: LocalModelsState,
  right: LocalModelsState,
): boolean => localModelsStateEquivalent(
  left,
  {
    ...right,
    models: right.models.map((model, index) =>
      alignUndisplayedHeadroom(left.models[index], model)),
  },
)
