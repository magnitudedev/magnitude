import { Option } from "effect"
import {
  localModelIsAvailable,
  localModelIsInstalled,
  localModelManagedAcquisition,
  type LocalModel,
  type LocalModelsState,
} from "@magnitudedev/sdk"
import type { ApplicationPage } from "./contracts"

/** A model on this computer that chats can use: installed and readable, not being removed, and not known not to fit. */
export const readyToChat = (model: LocalModel) =>
  localModelIsInstalled(model)
  && localModelIsAvailable(model)
  && model.acquisitionState._tag !== "Unavailable"
  && !Option.exists(localModelManagedAcquisition(model), (state) => state._tag === "Removing")
  && !(model.servingState._tag === "Assessed" && model.servingState.assessment._tag !== "Fits")

/**
 * Where the app opens: Chat once a model is ready, otherwise Models, which says none is downloaded yet and shows a
 * download in progress. The answer counts only once the service has finished finding the models on this computer.
 */
export const landingPage = (state: LocalModelsState): Option.Option<ApplicationPage> =>
  state.preparation.discovery.complete
    ? Option.some(state.models.some(readyToChat) ? "chat" : "models")
    : Option.none()
