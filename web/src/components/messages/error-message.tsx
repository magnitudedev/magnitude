/**
 * Error message — spec §9.3.10
 *
 * Box with red-tinted background, left border accent, [Error] tag,
 * message body, optional CTA (URL or Action button).
 */
import { type ReactNode } from "react"
import { Option } from "effect"
import type { ErrorDisplayMessage as ErrorType } from "@magnitudedev/sdk"
import { CopyButton, Timestamp } from "./shared"
type ErrorCtaValue = Option.Option.Value<ErrorType["cta"]>
function ErrorCta({ cta }: { cta: ErrorCtaValue }): ReactNode {
  if (cta.kind === "url") {
    return (
      <div className="[margin-top:6px] flex items-center [gap:8px]">
        <span className="font-sans text-[13px] text-slate-600 dark:text-slate-400">
          {cta.label}:
        </span>
        <a
          href={cta.url}
          target="_blank"
          rel="noopener noreferrer"
          className="text-blue-700 dark:text-blue-500 [text-decoration:underline] font-mono text-[13px]"
        >
          {cta.url}
        </a>
        <CopyButton text={cta.url} label="" />
      </div>
    )
  }
  // action
  return (
    <div className="[margin-top:6px]">
      <button className="bg-transparent hover:bg-red-300 hover:text-slate-900 dark:hover:bg-red-700 dark:hover:text-slate-200 border border-red-600 dark:border-red-500 rounded-[4px] [background:transparent] text-red-600 dark:text-red-500 font-mono text-[13px] [padding:4px_10px] cursor-pointer">
        {cta.label} ({cta.chord})
      </button>
    </div>
  )
}
export function ErrorMessage({ message }: { message: ErrorType }): ReactNode {
  const cta = Option.getOrNull(message.cta)
  return (
    <div>
      <div className="rounded-r border border-l-[3px] border-red-600 bg-red-200/40 px-3 py-2.5 dark:border-red-500 dark:bg-red-800/30">
        <div className="font-mono text-[13px] text-red-600 dark:text-red-500 font-semibold">
          [Error]
        </div>
        <div className="font-mono text-[14px] text-slate-900 dark:text-slate-200 whitespace-pre-wrap [margin-top:4px] leading-[1.5]">
          {message.message}
        </div>
        {cta && <ErrorCta cta={cta} />}
      </div>
      <div className="flex items-center [gap:8px] [margin-top:4px] [padding:0_2px]">
        <CopyButton text={message.message} />
        <Timestamp ts={message.timestamp} />
      </div>
    </div>
  )
}
