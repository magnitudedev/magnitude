import { CircleNotchIcon, PlusIcon } from "@phosphor-icons/react"
import { MagnitudeMark } from "../magnitude-mark"
import { LoadingRegion, SkeletonLine } from "../page-skeletons"
import { Skeleton } from "../ui/skeleton"

const SIDEBAR_WIDTHS = ["78%", "62%", "85%", "54%", "70%"] as const

/** Placeholder chat history rows in the sidebar while the service starts or chats load. */
export function ChatListSkeleton({ rows = 5 }: { rows?: number }) {
  return (
    <LoadingRegion label="Loading chats" className="chat-list" innerClassName="flex flex-col gap-px">
      {Array.from({ length: rows }, (_, index) => (
        <div key={index} className="flex h-8 min-w-0 items-center px-2.5">
          <SkeletonLine className="h-4 w-full text-[13px]" width={SIDEBAR_WIDTHS[index % SIDEBAR_WIDTHS.length]} />
        </div>
      ))}
    </LoadingRegion>
  )
}

/** Placeholder matching the model-picker trigger inside its 32px control row. */
export function ModelPickerSkeleton() {
  return (
    <span className="inline-flex h-8 items-center gap-2 px-2">
      <Skeleton className="size-4 shrink-0 rounded-full" />
      <Skeleton className="h-4 w-44" />
    </span>
  )
}

/**
 * The Chat page before the service or its models are ready, matching the loaded page's header and
 * composer geometry with non-interactive placeholders so nothing is prematurely clickable.
 */
export function ChatSkeleton({ startingService = false }: { startingService?: boolean }) {
  const label = startingService ? "Starting Magnitude…" : "Loading chat…"
  return (
    <section aria-label="Chat" className="flex min-h-0 flex-1 flex-col">
      <div className="mb-4 flex shrink-0 items-center justify-between gap-3">
        <h1 className="min-w-0 truncate border border-transparent font-heading text-[17px] font-semibold leading-7 tracking-tight">
          Chat
        </h1>
        <span
          aria-hidden="true"
          className="inline-flex h-8 shrink-0 select-none items-center gap-1.5 px-2.5 text-[13px] font-medium text-slate-600 opacity-50 dark:text-slate-400"
        >
          <PlusIcon className="size-4 shrink-0" />
          New chat
        </span>
      </div>
      <LoadingRegion label={label} className="flex min-h-0 flex-1 flex-col" innerClassName="flex min-h-0 flex-1 flex-col">
        <div className="chat-scroll min-h-0 flex-1 overflow-y-auto">
          <div className="flex h-full min-h-40 items-center justify-center">
            <div className="relative flex items-center justify-center">
              <MagnitudeMark className="size-12 opacity-20" />
              <div className="absolute top-full left-1/2 mt-3 flex -translate-x-1/2 items-center gap-2 whitespace-nowrap text-xs text-slate-500">
                <CircleNotchIcon className="size-3.5 shrink-0 motion-safe:animate-spin" />
                <span>{label}</span>
              </div>
            </div>
          </div>
        </div>
        <div className="mx-auto w-full max-w-[760px] shrink-0 pb-4 pt-3">
          <div className="rounded-xl border border-slate-300 bg-white p-3 dark:border-slate-700 dark:bg-slate-850">
            <div className="min-h-14 select-none px-1 py-1 text-sm leading-6 text-slate-500 dark:text-slate-600">
              Message…
            </div>
            <div className="mt-3 flex items-end justify-end gap-2">
              <ModelPickerSkeleton />
              <Skeleton className="size-8 shrink-0" />
            </div>
          </div>
        </div>
      </LoadingRegion>
    </section>
  )
}

/** Placeholder transcript shown while opening an existing conversation. */
export function ChatTranscriptSkeleton() {
  return (
    <LoadingRegion
      label="Loading conversation…"
      className="mx-auto max-w-[760px] pb-8 pt-7 pr-1"
      innerClassName="space-y-8"
    >
      <div className="ml-auto flex w-48 max-w-[88%] flex-col items-end">
        <div className="w-full rounded-2xl bg-slate-150 px-4 py-3 dark:bg-slate-800">
          <SkeletonLine className="h-6 text-sm" width="82%" />
        </div>
      </div>
      <div className="space-y-2">
        <SkeletonLine className="h-5 text-sm" width="92%" />
        <SkeletonLine className="h-5 text-sm" width="76%" />
        <SkeletonLine className="h-5 text-sm" width="48%" />
      </div>
    </LoadingRegion>
  )
}

/** Placeholder for Settings → Chat while servable models and defaults are still loading. */
export function ChatModelSettingsSkeleton() {
  return (
    <LoadingRegion label="Loading chat settings" innerClassName="space-y-6">
      <div className="rounded-lg border border-slate-300 bg-white px-4 py-3.5 dark:border-slate-750 dark:bg-slate-850">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="min-w-0 flex-1 space-y-1">
            <SkeletonLine className="h-5 text-sm" width="4.5rem" />
            <SkeletonLine className="h-4 text-xs" width="18rem" />
          </div>
          <Skeleton className="h-8 w-64 shrink-0" />
        </div>
      </div>
      <div className="rounded-lg border border-slate-300 bg-white px-4 py-4 dark:border-slate-750 dark:bg-slate-850">
        <div className="space-y-4">
          <div className="flex items-center justify-between gap-4">
            <SkeletonLine className="h-5 text-sm" width="8rem" />
            <Skeleton className="h-7 w-40 shrink-0" />
          </div>
          <SkeletonLine className="h-4 text-xs" width="14rem" />
          <Skeleton className="h-20 w-full" />
        </div>
      </div>
    </LoadingRegion>
  )
}
