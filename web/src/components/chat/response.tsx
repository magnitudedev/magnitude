import { memo, useMemo, useState } from "react"
import { DateTime, Option } from "effect"
import {
  type ChatAssistantMessage,
  responseExceedsContext,
  responseMetrics,
  responseStatusText,
} from "@magnitudedev/sdk"
import {
  ArrowCounterClockwiseIcon,
  CaretDownIcon,
  PlusIcon,
  CaretRightIcon,
} from "@phosphor-icons/react"
import Markdown, { type Components } from "react-markdown"
import remarkGfm from "remark-gfm"
import { CopyText, IconButton, SentAt, ThinkingFor, number, seconds } from "./shared"
import { ChatCodeBlock } from "./code-block"
import { rehypeLongTokens } from "./long-tokens"
import { FailureDetails } from "../failure-details"
/** A response's parts of one kind, joined in order. */
const partsText = (response: ChatAssistantMessage, part: "text" | "reasoning") =>
  response.content.flatMap((p) => (p._tag === part ? [p.text] : [])).join("")

export function ChatResponse({
  response: g,
  retry,
  newChat,
  notify,
}: {
  response: ChatAssistantMessage
  retry?: () => void
  newChat?: () => void
  notify: (text: string) => void
}) {
  const [details, setDetails] = useState(false)
  const m = responseMetrics(g)
  const content = partsText(g, "text")
  const reasoning = partsText(g, "reasoning")
  const running = Option.isNone(g.end)
  const thinkingSince = running && g.content.at(-1)?._tag === "reasoning"
    ? g.content.find((part) => part._tag === "reasoning")?.time.started
    : undefined
  const { label, problem: failure, detail } = responseStatusText(g)
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
      Option.map(m.prefillTokensPerSecond, (n) => `${number(n)} tok/s`),
    ],
    ["First answer", Option.map(m.firstAnswerMs, seconds)],
    ["Total", Option.map(m.totalMs, seconds)],
  ]
  return (
    <article aria-label="Assistant response" className="chat-hover-group">
      {reasoning && (
        <details className="mb-3">
          <summary className="cursor-pointer py-1 text-xs text-slate-500 dark:text-slate-400">
            {thinkingSince !== undefined
              ? <ThinkingFor started={DateTime.toEpochMillis(thinkingSince)} />
              : running && !content
              ? "Thinking"
              : Option.match(m.thoughtForMs, {
                  onNone: () => "Thought process",
                  onSome: (ms) => `Thought for ${seconds(ms)}`,
                })}
          </summary>
          <div className="mt-2 whitespace-pre-wrap break-words border-l-2 border-slate-200 pl-3 text-[13px] leading-6 text-slate-500 dark:border-slate-700 dark:text-slate-400">
            {reasoning}
          </div>
        </details>
      )}
      {content && <ChatMarkdown content={content} notify={notify} />}
      {!running && (
        <>
          {Option.isSome(failure) && (
            <p
              role="alert"
              className="mt-3 text-xs leading-5 text-red-600 dark:text-red-400"
            >
              {failure.value}
            </p>
          )}
          {Option.isSome(detail) && (
            <FailureDetails text={detail.value} notify={notify} />
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
                ...Option.toArray(label),
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
            <div className="flex items-center gap-1">
              <span className="chat-message-actions">
                {Option.match(g.end, { onNone: () => null, onSome: (end) => <SentAt time={DateTime.toEpochMillis(end.time)} /> })}
                {content && <CopyText text={content} label="Copy response" notify={notify} />}
              </span>
              {retry && responseExceedsContext(g) && newChat ? (
                <IconButton label="New chat" onClick={newChat}>
                  <PlusIcon />
                </IconButton>
              ) : (
                retry && (
                  <IconButton label="Retry response" onClick={retry}>
                    <ArrowCounterClockwiseIcon />
                  </IconButton>
                )
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
  notify,
}: {
  content: string
  notify: (text: string) => void
}) {
  // Stable renderer identities preserve copy feedback while new tokens arrive.
  const components = useMemo<Components>(
    () => ({
      pre: ({ node, children }) => {
        const code = node?.children.find(
          (child) => child.type === "element" && child.tagName === "code"
        )
        if (!code || code.type !== "element") return <pre>{children}</pre>
        // Markdown adds one display newline. Remove only that one, preserving
        // indentation, trailing spaces and intentional blank lines when copying.
        const text = code.children
          .flatMap((child) => (child.type === "text" ? [child.value] : []))
          .join("")
          .replace(/\n$/, "")
        const classes = code.properties.className
        const language = Array.isArray(classes)
          ? String(
              classes.find((name) => String(name).startsWith("language-")) ?? ""
            ).slice(9)
          : ""
        return <ChatCodeBlock code={text} language={language} notify={notify} />
      },
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
    }),
    [notify]
  )
  return (
    <div className="chat-prose text-sm leading-6">
      <Markdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeLongTokens]} components={components}>
        {content}
      </Markdown>
    </div>
  )
})
