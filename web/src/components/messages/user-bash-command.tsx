import type { ReactNode } from "react"
import type { UserBashCommandMessage } from "@magnitudedev/sdk"
import { CopyButton, Timestamp } from "./shared"
export function UserBashCommand({
  message,
}: {
  message: UserBashCommandMessage
}): ReactNode {
  const output = [message.stdout, message.stderr].filter(Boolean).join("\n")
  const failed = message.exitCode !== 0
  return (
    <div>
      <div className="rounded-r border border-l-[3px] border-slate-300 border-l-orange-700 px-[11px] py-2 font-mono text-[13px] dark:border-slate-750 dark:border-l-orange-400">
        <div className="text-slate-900 dark:text-slate-200 font-semibold">
          <span className="text-orange-700 dark:text-orange-400">$ </span>
          {message.command}
          <span
            className={`${
              failed
                ? "text-red-600 dark:text-red-500"
                : "text-green-700 dark:text-green-500"
            }  [margin-left:8px]`}
          >
            {failed ? `Exit ${message.exitCode}` : "✓"}
          </span>
        </div>
        {output && (
          <pre
            className={`${
              failed
                ? "text-red-600 dark:text-red-500"
                : "text-slate-600 dark:text-slate-400"
            }  [margin:6px_0_0] whitespace-pre-wrap [word-break:break-word]`}
          >
            {output}
          </pre>
        )}
      </div>
      <div className="flex items-center [gap:8px] [margin-top:4px] [padding:0_2px]">
        <CopyButton text={message.command} />
        <Timestamp ts={message.timestamp} />
      </div>
    </div>
  )
}
