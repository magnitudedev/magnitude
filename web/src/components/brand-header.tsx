import type { ReactNode } from "react"
import { MagnitudeMark } from "./magnitude-mark"

/** The Magnitude mark and name where the sidebar puts them, for screens shown without the sidebar. */
export function BrandHeader({ children }: { children?: ReactNode }) {
  return <header className="mt-14 flex h-8 items-center justify-between pl-7 pr-8">
    <span className="flex items-center gap-3 font-heading text-base font-semibold"><MagnitudeMark className="h-8 w-8 shrink-0" />Magnitude</span>
    {children}
  </header>
}

/** Before the app knows whether it opens on onboarding or the app itself: only what both share. */
export function BootScreen() {
  return <div aria-busy="true" className="h-screen bg-slate-50 font-sans text-slate-900 dark:bg-slate-925 dark:text-slate-200">
    <span role="status" className="sr-only">Opening Magnitude</span>
    <BrandHeader />
  </div>
}
