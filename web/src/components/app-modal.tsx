import type { ReactNode } from "react"
import { XIcon } from "@phosphor-icons/react"
import { Button } from "./ui/button"
import { Dialog, DialogClose, DialogContent, DialogTitle } from "./ui/dialog"

/** The overlay Settings opens in over the page: onboarding's size, its title and close, and a body. */
export function AppModal({ title, onClose, children }: { title: ReactNode; onClose: () => void; children: ReactNode }) {
  return <Dialog open onOpenChange={open => { if (!open) onClose() }}>
    <DialogContent showCloseButton={false} className="flex h-[88vh] flex-col gap-0 overflow-hidden p-0 sm:max-w-4xl">
      <div className="flex shrink-0 items-center justify-between gap-4 border-b border-slate-200 px-6 py-4 dark:border-slate-750">
        <DialogTitle className="truncate font-heading text-lg font-semibold">{title}</DialogTitle>
        <DialogClose render={<Button variant="ghost" size="icon-sm" aria-label="Close" />}><XIcon /></DialogClose>
      </div>
      <div className="flex min-h-0 flex-1 flex-col">{children}</div>
    </DialogContent>
  </Dialog>
}

/** A modal's scrolling content area. */
export function ModalBody({ children }: { children: ReactNode }) {
  return <div className="min-h-0 flex-1 overflow-y-auto px-6 pb-6 pt-5">{children}</div>
}
