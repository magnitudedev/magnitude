import { XIcon } from "@phosphor-icons/react"
import type { Result } from "@effect-atom/atom-react"
import { errorText } from "./shared"

/**
 * The one error line above the composer: the first failure among `results`, in the order given, or none while a
 * request is pending. Several results often fail for the same reason, such as the service being down, so one line says
 * it once, and not at all when the chat's own notice above (`shown`) already says the same.
 */
export const composerError = (pending: boolean, results: readonly Result.Result<unknown, unknown>[], shown: string | null = null): string | null => {
  if (pending) return null
  const found = results.reduce<string | null>((first, result) => first ?? errorText(result), null)
  return found === shown ? null : found
}

/** A notice or error above the composer, which the person can dismiss. */
export function ComposerAlert({ text, onDismiss }: { text: string; onDismiss: () => void }) {
  return <div role="alert" className="mb-3 flex items-start justify-between gap-3 text-xs leading-5 text-red-600 dark:text-red-400">
    <span>{text}</span>
    <button type="button" aria-label="Dismiss notice" onClick={onDismiss}>
      <XIcon />
    </button>
  </div>
}
