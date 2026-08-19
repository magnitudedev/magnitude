import type { ReactNode } from "react"

import type { SettingsTab } from "../state/web-atoms"
import { ArchivedChatsView } from "./archived-chats-view"
import { ModelSettingsCenter } from "./model-center"

export function SettingsCenter({ tab }: { readonly tab: SettingsTab }): ReactNode {
  return tab === "archived"
    ? <div className="min-h-0 min-w-0 flex-1 overflow-auto"><ArchivedChatsView /></div>
    : <ModelSettingsCenter tab={tab} />
}
