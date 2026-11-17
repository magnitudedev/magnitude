import { memo, useMemo } from "react"
import { DateTime, Option } from "effect"
import {
  type ChatAssistantMessage,
  responseExceedsContext,
  responseMetrics,
  responseStatusText,
} from "@magnitudedev/sdk"
import { ArrowCounterClockwiseIcon, PlusIcon } from "@phosphor-icons/react"
import Markdown, { type Components } from "react-markdown"
import remarkGfm from "remark-gfm"
import { CopyText, SentAt, ThinkingFor, number, seconds, tokens } from "./shared"
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
  const m = responseMetrics(g)
  const content = partsText(g, "text")
  const reasoning = partsText(g, "reasoning")
  const running = Option.isNone(g.end)
  const thinkingSince = running && g.content.at(-1)?._tag === "reasoning"
    ? g.content.find((part) => part._tag === "reasoning")?.time.started
    : undefined
  const { label, problem: failure, detail } = responseStatusText(g)
  // Every measurement the engine reported, condensed onto one line; a bare number gets a word only where it'd be ambiguous.
  const at = (speed: Option.Option<number>) => Option.match(speed, { onNone: () => "", onSome: (n) => ` at ${Math.round(n).toLocaleString()} tok/s` })
  const metrics = [
    Option.map(m.inputTokens, (n) => `Prefill ${tokens(n)}${Option.match(Option.filter(m.cachedTokens, (cached) => cached > 0), { onNone: () => "", onSome: (cached) => ` (${number(cached)} cached)` })}${at(m.prefillTokensPerSecond)}`),
    Option.map(m.outputTokens, (n) => `Decode ${tokens(n)}${at(m.tokensPerSecond)}`),
    Option.map(m.ttftMs, (n) => `TTFT ${seconds(n)}`),
    Option.map(m.totalMs, (n) => `${seconds(n)} total`),
    label,
  ].flatMap(Option.toArray)
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
          {metrics.length > 0 && (
            <p className="mt-4 text-xs tabular-nums text-slate-500 dark:text-slate-400">{metrics.join(" · ")}</p>
          )}
          <div className="mt-1 flex items-center">
            <span className="chat-message-actions">
              {Option.match(g.end, { onNone: () => null, onSome: (end) => <SentAt time={DateTime.toEpochMillis(end.time)} /> })}
              {content && <CopyText text={content} label="Copy response" notify={notify} />}
              {retry && responseExceedsContext(g) && newChat ? (
                <button type="button" className="chat-message-action" aria-label="New chat" title="New chat" onClick={newChat}>
                  <PlusIcon />
                </button>
              ) : (
                retry && (
                  <button type="button" className="chat-message-action" aria-label="Retry response" title="Retry response" onClick={retry}>
                    <ArrowCounterClockwiseIcon />
                  </button>
                )
              )}
            </span>
          </div>
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
      // A list's numbers sit in its own padding, so it is as wide as its widest number ("138.") and never cut off.
      ol: ({ node, children, start }) => {
        const items = node?.children.filter((child) => child.type === "element" && child.tagName === "li").length ?? 0
        const digits = String(Math.max(1, (start ?? 1) + items - 1)).length
        return <ol start={start} style={{ paddingLeft: `calc(${digits}ch + 1em)` }}>{children}</ol>
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
