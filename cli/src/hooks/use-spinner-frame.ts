import { animationStep } from '@magnitudedev/client-common'
import { useAnimationStep } from './use-animation-time'

const SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"] as const

const SPINNER_FRAME_MS = 80

export const spinnerFrameAt = (timeMs: number): string =>
  SPINNER_FRAMES[animationStep(timeMs, SPINNER_FRAME_MS) % SPINNER_FRAMES.length]!

export const spinnerFrameForStep = (step: number): string =>
  SPINNER_FRAMES[step % SPINNER_FRAMES.length]!

export function useSpinnerFrame(active = true): string {
  return spinnerFrameForStep(useAnimationStep(active, SPINNER_FRAME_MS))
}
