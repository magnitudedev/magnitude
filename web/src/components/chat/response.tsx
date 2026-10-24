import { memo, useState } from "react"
import { Option } from "effect"
import type { ChatGeneration } from "@magnitudedev/sdk"
import {
  ArrowCounterClockwiseIcon,
  CaretDownIcon,
  CaretRightIcon,
  CopyIcon,
  CheckIcon,
} from "@phosphor-icons/react"
import Markdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { IconButton, number, seconds } from "./shared"
export function ChatResponse({
  generation: g,
  retry,
  notify,
}: {
  generation: ChatGeneration
  retry?: () => void
  notify: (text: string) => void
}) {
  const [details, setDetails] = useState(false)
  const [copied, setCopied] = useState(false)
  const m = g.metrics
  const metrics = [
    Option.map(m.tokensPerSecond, (n) => `${number(n)} tok/s`),
    Option.map(m.ttftMs, (n) => `TTFT ${seconds(n)}`),
    Option.map(m.outputTokens, (n) => `${number(n)} tokens`),
  ].flatMap(Option.toArray)
  const labels: [string, Option.Option<string>][] = [
    ["Input", Option.map(m.inputTokens, (n) => `${number(n)} tokens`)],
    ["Cached", Option.map(m.cachedTokens, (n) => `${number(n)} tokens`)],
    [
      "Prefill",
      Option.map(m.promptTokensPerSecond, (n) => `${number(n)} tok/s`),
    ],
    ["First answer", Option.map(m.firstAnswerMs, seconds)],
    ["Total", Option.map(m.totalMs, seconds)],
  ]
  return (
    <article aria-label="Assistant response">
      {g.reasoning && (
        <details className="mb-3">
          <summary className="cursor-pointer py-1 text-xs text-slate-500 dark:text-slate-400">
            {g.status === "running" && !g.content
              ? "Thinking"
              : Option.match(m.reasoningMs, {
                  onNone: () => "Thought process",
                  onSome: (ms) => `Thought for ${seconds(ms)}`,
                })}
          </summary>
          <div className="mt-2 whitespace-pre-wrap break-words border-l-2 border-slate-200 pl-3 text-[13px] leading-6 text-slate-500 dark:border-slate-700 dark:text-slate-400">
            {g.reasoning}
          </div>
        </details>
      )}
      {g.content && <ChatMarkdown content={g.content} />}
      {g.status !== "running" && (
        <>
          {Option.isSome(g.error) && (
            <p
              role="alert"
              className="mt-3 text-xs leading-5 text-red-600 dark:text-red-400"
            >
              {g.error.value}
            </p>
          )}
          <div className="mt-4 flex flex-wrap items-center justify-between gap-2">
            <button
              type="button"
              aria-label="Response metrics"
              aria-expanded={details}
              onClick={() => setDetails(!details)}
              className="flex flex-wrap items-center gap-2 rounded py-1 text-xs tabular-nums text-slate-500 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-200"
            >
              {[
                ...metrics,
                ...(g.status === "stopped"
                  ? ["Stopped"]
                  : g.status === "length"
                  ? ["Output limit reached"]
                  : g.status === "failed"
                  ? ["Failed"]
                  : g.status === "interrupted"
                  ? ["Interrupted"]
                  : []),
              ].map((metric, i) => (
                <span key={metric} className="inline-flex items-center gap-2">
                  {i > 0 && (
                    <span className="text-slate-300 dark:text-slate-700">
                      ·
                    </span>
                  )}
                  {metric}
                </span>
              ))}
              {details ? <CaretDownIcon /> : <CaretRightIcon />}
            </button>
            <div className="flex items-center">
              <IconButton
                label={copied ? "Copied" : "Copy response"}
                disabled={!g.content}
                onClick={() => {
                  void navigator.clipboard.writeText(g.content).then(
                    () => setCopied(true),
                    () => notify("Copy is unavailable in this browser.")
                  )
                }}
              >
                {copied ? <CheckIcon /> : <CopyIcon />}
              </IconButton>
              {retry && (
                <IconButton label="Retry response" onClick={retry}>
                  <ArrowCounterClockwiseIcon />
                </IconButton>
              )}
            </div>
          </div>
          {details && (
            <dl className="mt-1 flex flex-wrap gap-x-5 gap-y-2 pb-1 text-xs leading-5 tabular-nums text-slate-500 dark:text-slate-400">
              {labels.flatMap(([label, item]) =>
                Option.isSome(item)
                  ? [
                      <div
                        key={label}
                        className="inline-flex items-baseline gap-1.5"
                      >
                        <dt>{label}</dt>
                        <dd className="text-slate-700 dark:text-slate-300">
                          {item.value}
                        </dd>
                      </div>,
                    ]
                  : []
              )}
            </dl>
          )}
        </>
      )}
    </article>
  )
}

const ChatMarkdown = memo(function ChatMarkdown({
  content,
}: {
  content: string
}) {
  return (
    <div className="chat-prose text-sm leading-6">
      <Markdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ children, href }) => (
            <a href={href} target="_blank" rel="noreferrer noopener">
              {children}
            </a>
          ),
          img: ({ alt }) => (
            <span className="text-slate-500">
              {alt ? `[Image: ${alt}]` : "[Image]"}
            </span>
          ),
        }}
      >
        {content}
      </Markdown>
    </div>
  )
})
