import { useMemo } from "react"
import { createPortal } from "react-dom"
import { Atom, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import type { HostNotice } from "@magnitudedev/client-common"
import { useSession } from "../session"
import { ErrorNotice, NoticeAction } from "./error-notice"

/** Above every dialog layer (dialogs use z-50), so a notice raised while the Add dialog is open, like an invalid magnitude:// link, stays visible and dismissable. */
export const hostNoticeLayer = "z-[60]"

export function HostNoticeStack({ notices, onDismiss }: { notices: readonly HostNotice[]; onDismiss: (notice: HostNotice) => void }) {
  if (notices.length === 0) return null
  return <div data-host-notices="" className={`pointer-events-auto fixed bottom-4 left-4 ${hostNoticeLayer} flex w-96 max-w-[calc(100vw-2rem)] flex-col gap-2`}>
    {notices.map((notice, index) => <div key={index} className="rounded-lg bg-white shadow-md dark:bg-slate-850">
      <ErrorNotice title={notice.title} description={notice.description} actions={<NoticeAction onClick={() => onDismiss(notice)}>Dismiss</NoticeAction>} />
    </div>)}
  </div>
}

/** Electron main's notices, portaled to the document body beside the dialog portals and layered above them. */
export function HostNotices() {
  const session = useSession()
  const notices = useAtomValue(session.notices)
  const dismiss = useAtomSet(useMemo(() => Atom.fn((notice: HostNotice) => session.dismissNotice(notice)), [session]))
  return createPortal(<HostNoticeStack notices={notices} onDismiss={dismiss} />, document.body)
}
