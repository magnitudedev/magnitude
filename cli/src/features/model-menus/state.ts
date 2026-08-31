import { Option, Schema } from "effect"
import {
  LocalModelsStateSchema,
  localModelServingState,
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
  if (previousModel === undefined) return nextModel
  const previousServing = Option.getOrUndefined(localModelServingState(previousModel))
  const nextServing = Option.getOrUndefined(localModelServingState(nextModel))
  if (previousServing?._tag !== "Assessed"
    || previousServing.assessment._tag !== "Fits"
    || nextServing?._tag !== "Assessed"
    || nextServing.assessment._tag !== "Fits") return nextModel
  const previousHeadroomState = previousServing.assessment.memory.currentHeadroomState
  const nextHeadroomState = nextServing.assessment.memory.currentHeadroomState
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
  if (nextModel._tag === "Catalog") {
    const servingState = nextModel.servingState
    if (servingState._tag !== "Assessed"
      || servingState.assessment._tag !== "Fits"
      || !("rankingScores" in servingState)) {
      return nextModel
    }
    const assessment = servingState.assessment
    return {
      ...nextModel,
      servingState: {
        ...servingState,
        assessment: {
          ...assessment,
          memory: {
            ...assessment.memory,
            currentHeadroomState: alignedCurrentHeadroomState,
          },
        },
      },
    }
  }
  switch (nextModel.state._tag) {
    case "Ready": {
      const discoveredServing = nextModel.state.servingState
      if (discoveredServing._tag !== "Assessed"
        || discoveredServing.assessment._tag !== "Fits") return nextModel
      return {
        ...nextModel,
        state: {
          ...nextModel.state,
          servingState: {
            ...discoveredServing,
            assessment: {
              ...discoveredServing.assessment,
              memory: {
                ...discoveredServing.assessment.memory,
                currentHeadroomState: alignedCurrentHeadroomState,
              },
            },
          },
        },
      }
    }
    case "Unavailable":
    case "Ambiguous": return nextModel
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
