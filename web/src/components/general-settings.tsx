import type { ReactNode } from "react"
import { Switch } from "@/components/ui/switch"
import { useShowThinkingPreference } from "@/stores/conversation-preferences"

export function GeneralSettings(): ReactNode {
  const [showThinking, setShowThinking] = useShowThinkingPreference()

  return (
    <div className="mx-auto flex min-h-full w-full max-w-[1040px] flex-col px-7 py-8 max-[700px]:px-4">
      <header className="mb-7">
        <h1 className="font-heading text-[28px] font-semibold text-slate-900 dark:text-slate-100">
          General
        </h1>
      </header>
      <section
        aria-labelledby="conversation-settings-title"
        className="overflow-hidden rounded-lg border border-slate-300 bg-white dark:border-slate-750 dark:bg-slate-850"
      >
        <div className="border-b border-slate-200 px-4 py-3 dark:border-slate-800">
          <h2
            id="conversation-settings-title"
            className="font-heading text-[18px] text-slate-900 dark:text-slate-100"
          >
            Conversation
          </h2>
        </div>
        <label className="flex min-h-[72px] cursor-pointer items-center justify-between gap-8 px-4 py-3.5">
          <span className="flex min-w-0 flex-col gap-1">
            <span className="font-sans text-[13px] font-semibold text-slate-900 dark:text-slate-200">
              Show thinking
            </span>
            <span className="font-sans text-[12px] leading-5 text-slate-500">
              Show model reasoning inline with the conversation.
            </span>
          </span>
          <Switch
            checked={showThinking}
            onCheckedChange={setShowThinking}
            aria-label="Show thinking"
          />
        </label>
      </section>
    </div>
  )
}
