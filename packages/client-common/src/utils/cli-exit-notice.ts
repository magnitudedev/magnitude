import { Option } from "effect"
import type { AcnClientCloseResult } from "@magnitudedev/sdk"

export const CLI_EXIT_OBSERVATION_FALLBACK =
  "Magnitude may still have background processes running.\n" +
  "Run `magnitude server stop` to stop the background service."

/** A healthy close needs no residency warning; the background service owns lifetime. */
export const deriveCliExitNotice = (observation: AcnClientCloseResult): Option.Option<string> =>
  Option.isSome(observation) ? Option.none() : Option.some(CLI_EXIT_OBSERVATION_FALLBACK)
