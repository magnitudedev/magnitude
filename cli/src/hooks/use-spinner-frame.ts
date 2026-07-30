import { useAnimationTick } from "./use-animation-tick"

const SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"] as const

export const spinnerFrameForTick = (tick: number): string =>
  SPINNER_FRAMES[tick % SPINNER_FRAMES.length]!

export function useSpinnerFrame(active = true): string {
  return spinnerFrameForTick(useAnimationTick(active))
}
