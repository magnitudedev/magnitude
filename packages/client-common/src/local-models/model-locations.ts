import type { ModelLocation, ModelLocationPath } from "@magnitudedev/sdk"

/**
 * The directories a location's row shows: those found or unreadable, in the location's order, or
 * every directory Magnitude checked when none exists.
 */
export const shownModelLocationPaths = (location: ModelLocation): readonly [ModelLocationPath, ...ModelLocationPath[]] => {
  const [first, ...rest] = location.paths.filter(({ state }) => state !== "NotFound")
  return first === undefined ? location.paths : [first, ...rest]
}
