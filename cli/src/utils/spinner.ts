import { animationStep } from "@magnitudedev/client-common"

const SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"] as const

export const SPINNER_FRAME_MS = 80

export const spinnerFrameAt = (timeMs: number): string =>
  SPINNER_FRAMES[animationStep(timeMs, SPINNER_FRAME_MS) % SPINNER_FRAMES.length]!

export const spinnerFrameForStep = (step: number): string =>
  SPINNER_FRAMES[step % SPINNER_FRAMES.length]!
