/**
 * The mark beside the status title: the service failing or starting, or, ready, a running dot beside the model that is
 * active. Beside "No model loaded" a running dot would read as a model running, so there it shows none.
 */
export const serviceIndicator = ({ ready, unavailable, modelActive }: { ready: boolean; unavailable: boolean; modelActive: boolean }): "failed" | "ready" | "starting" | null =>
  unavailable ? "failed" : !ready ? "starting" : modelActive ? "ready" : null
