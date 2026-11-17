import { Option } from "effect"

/**
 * Why a message or retry can't be sent because of its model, when that is why: no model chosen, the chat's model no
 * longer served, or that model's settings out of range. Anything else (a response running, an empty message) needs
 * no telling.
 */
export const submitBlocked = (state: { readonly chosen: boolean; readonly available: boolean; readonly settingsValid: boolean }): Option.Option<string> =>
  !state.chosen ? Option.some("Choose a model beside Send, then try again.")
    : !state.available ? Option.some("This chat’s model isn’t available anymore. Choose another model beside Send, then try again.")
    : !state.settingsValid ? Option.some("This model’s settings are out of range. Check them in Settings → Chat, then try again.")
    : Option.none()
