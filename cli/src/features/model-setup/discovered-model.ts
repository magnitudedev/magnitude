import { homedir } from "node:os"
import type { LocalModel } from "@magnitudedev/sdk"

type DiscoveredLocalModel = Extract<LocalModel, { readonly _tag: "Discovered" }>
type LocatedDiscoveryState = Exclude<
  DiscoveredLocalModel["state"],
  { readonly _tag: "Ambiguous" }
>

export const discoveredModelLocation = (state: LocatedDiscoveryState): string => {
  const path = state.installation.primaryPath
  const home = homedir()
  const separator = path[home.length]
  return path === home
    ? "~"
    : path.startsWith(home) && (separator === "/" || separator === "\\")
      ? `~${path.slice(home.length)}`
      : path
}
