import { useId, useMemo, useRef, type ReactNode } from "react"
import { createPortal } from "react-dom"
import { Atom, useAtomMount } from "@effect-atom/atom-react"
import { Effect } from "effect"
import { X } from "@phosphor-icons/react"
import { createFocusTrapHandler } from "../utils/focus-trap"

export function Dialog({
  title,
  description,
  children,
  onDismiss,
  size = "medium",
}: {
  readonly title: string
  readonly description?: string
  readonly children: ReactNode
  readonly onDismiss: () => void
  readonly size?: "small" | "medium"
}): ReactNode {
  const dialogRef = useRef<HTMLElement | null>(null)
  const previouslyFocusedRef = useRef<HTMLElement | null>(
    typeof document === "undefined" ? null : document.activeElement as HTMLElement | null,
  )
  const focusRestorationAtom = useMemo(() => Atom.make(
    Effect.addFinalizer(() => Effect.sync(() => {
      const previous = previouslyFocusedRef.current
      if (previous?.isConnected) previous.focus()
    })),
  ), [])
  useAtomMount(focusRestorationAtom)
  const titleId = useId()
  const descriptionId = useId()
  if (typeof document === "undefined") return null
  return createPortal(
    <div
      className="fixed inset-0 z-[120] flex items-center justify-center bg-black/45 p-5 dark:bg-black/70"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onDismiss()
      }}
      onKeyDown={(event) => {
        if (event.key === "Escape") onDismiss()
      }}
      role="presentation"
    >
      <section
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={description ? descriptionId : undefined}
        tabIndex={-1}
        onKeyDown={createFocusTrapHandler(dialogRef, onDismiss)}
        className={`${size === "small" ? "max-w-[400px]" : "max-w-[520px]"} w-full rounded-xl border border-slate-200 bg-white shadow-xl dark:border-slate-750 dark:bg-slate-800`}
      >
        <header className="flex items-start gap-4 px-5 pb-3 pt-5">
          <div className="min-w-0 flex-1">
            <h2 id={titleId} className="font-mono text-[15px] font-semibold text-slate-900 dark:text-slate-100">
              {title}
            </h2>
            {description ? (
              <p id={descriptionId} className="mt-1.5 font-sans text-[13px] leading-5 text-slate-600 dark:text-slate-400">
                {description}
              </p>
            ) : null}
          </div>
          <button
            type="button"
            onClick={onDismiss}
            aria-label="Close dialog"
            className="-mr-1 -mt-1 flex size-8 shrink-0 items-center justify-center rounded-md border-0 bg-transparent text-slate-500 hover:bg-slate-100 hover:text-slate-800 dark:hover:bg-slate-750 dark:hover:text-slate-200"
          >
            <X size={16} />
          </button>
        </header>
        {children}
      </section>
    </div>,
    document.body,
  )
}

export function DialogActions({ children }: { readonly children: ReactNode }): ReactNode {
  return (
    <footer className="flex items-center justify-end gap-2 border-t border-slate-200 px-5 py-4 dark:border-slate-750">
      {children}
    </footer>
  )
}

export const dialogSecondaryButton =
  "h-9 rounded-md border border-slate-300 bg-white px-3.5 font-sans text-[13px] font-medium text-slate-700 hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-50 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-300 dark:hover:bg-slate-750"

export const dialogPrimaryButton =
  "h-9 rounded-md border border-blue-700 bg-blue-700 px-3.5 font-sans text-[13px] font-semibold text-white hover:border-blue-800 hover:bg-blue-800 disabled:cursor-not-allowed disabled:opacity-50 dark:border-blue-500 dark:bg-blue-500 dark:text-slate-925 dark:hover:border-blue-400 dark:hover:bg-blue-400"
