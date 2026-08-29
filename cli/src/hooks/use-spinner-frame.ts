import { useAnimationStep } from './use-animation-time'
import {
  SPINNER_FRAME_MS,
  spinnerFrameForStep,
} from '../utils/spinner'

export { spinnerFrameAt, spinnerFrameForStep } from '../utils/spinner'

export function useSpinnerFrame(active = true): string {
  return spinnerFrameForStep(useAnimationStep(active, SPINNER_FRAME_MS))
}
