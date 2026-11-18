/**
 * What an inference engine error means to the person who saw it, by its code: what happened and what to do. A code
 * says what failed, not when (a worker can exit while a model loads or while it answers) nor always why, so each
 * sentence claims no more than its code does. The engine's own message is diagnostic text (a worker's exit, its
 * backtrace), kept for reports and never shown as this.
 */
const sentences: ReadonlyMap<string, string> = new Map(Object.entries({
  context_length_exceeded: "This chat is longer than the context this computer serves this model at. Start a new chat to continue.",
  context_exceeds_memory: "This chat needs more memory than this computer can give the model. Start a new chat, or choose a smaller model.",
  insufficient_memory: "There wasn’t enough free memory for this model. Close other apps or stop another model, then try again.",
  memory_shortage: "There wasn’t enough free memory for this model. Close other apps or stop another model, then try again.",
  memory_observation_unavailable: "Magnitude couldn’t check this computer’s free memory. Try again.",
  model_does_not_fit: "This model doesn’t fit on this computer. Choose a smaller model.",
  model_unloaded: "The model was unloaded before it finished. Try again.",
  model_instance_stopped: "The model was stopped before it finished. Try again.",
  model_not_found: "This model isn’t available anymore. Choose another model.",
  model_not_ready: "The model isn’t ready yet. Wait for it to finish loading, then try again.",
  model_busy: "The model is busy with another change. Try again in a moment.",
  worker_exited: "The model’s engine stopped unexpectedly. Try again, or choose another model.",
  worker_unresponsive: "The model’s engine stopped responding. Try again, or choose another model.",
  worker_spawn_failed: "Magnitude couldn’t start the model’s engine. Try again, or choose another model.",
  model_load_failed: "The model couldn’t be loaded. Try again, or choose another model.",
  weight_read_failed: "Magnitude couldn’t read the model’s files. Check the disk they are on, then try again.",
  integrity_failed: "The model’s files failed their integrity check. Remove the model and download it again.",
  invalid_model_package: "The model’s files can’t be used. Remove the model and download it again.",
  invalid_model_configuration: "The model’s settings can’t be used. Choose another model.",
  unsupported_model: "Magnitude can’t run this model yet. Choose another model.",
  device_lost: "The device running the model stopped working. Try again; if it keeps happening, restart Magnitude.",
  device_unavailable: "The device the model runs on isn’t available. Restart Magnitude, then try again.",
  prices_pending: "Magnitude is still measuring this computer. Try again in a moment.",
  backend_error: "The inference engine hit an error. Try again, or choose another model.",
  invalid_request: "The inference engine refused this request as invalid. Try again, or check this model’s settings in Settings → Chat.",
  request_cancelled: "The request was cancelled. Try again.",
}))

export const engineFailureSentence = (code: string): string =>
  sentences.get(code) ?? `The inference engine reported an error (${code}). Try again, or choose another model.`
