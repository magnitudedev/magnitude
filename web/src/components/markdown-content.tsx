/**
 * MarkdownContent — spec §14.1
 *
 * react-markdown + remark-gfm wrapper with shiki code highlighting.
 * Overrides for code blocks, links, headings, blockquotes, lists, tables.
 * Streaming cursor support.
 *
 * Uses useSyncExternalStore for the shared Shiki highlighter — no useEffect.
 */
import {
  memo,
  useSyncExternalStore,
  useMemo,
  useState,
  type ReactNode,
} from "react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import type { Components } from "react-markdown"
import { Copy, Check } from "lucide-react"
import {
  subscribeShiki,
  getShikiSnapshot,
  highlightCode,
} from "../stores/shiki-store"

/** Highlight a code block — returns highlighted HTML or null if not yet loaded */
function useCodeHighlight(code: string, lang: string): string | null {
  // Subscribe to shiki store — re-renders when highlighter loads
  const highlighter = useSyncExternalStore(subscribeShiki, getShikiSnapshot)
  return useMemo(() => highlightCode(code, lang), [code, highlighter, lang])
}

/** Copy button with feedback state */
function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)
  return (
    <button
      className={`${
        copied ? "text-green-700 dark:text-green-500" : "text-slate-500"
      }  code-block-copy [background:transparent] border-0 cursor-pointer flex items-center [padding:2px]`}
      aria-label="Copy code"
      onClick={(e) => {
        e.stopPropagation()
        navigator.clipboard?.writeText(text)
        setCopied(true)
        setTimeout(() => setCopied(false), 2000)
      }}
    >
      {copied ? <Check size={14} /> : <Copy size={14} />}
    </button>
  )
}

/** Code block component with shiki highlighting */
function CodeBlock({ code, lang }: { code: string; lang: string }) {
  const highlighted = useCodeHighlight(code, lang)
  return (
    <div className="code-block [margin:8px_0_12px] rounded-[4px] overflow-hidden">
      <div className="code-block-header flex items-center justify-between bg-white dark:bg-slate-850 border-b border-b-slate-300 dark:border-b-slate-750 [padding:4px_10px] font-sans text-[11px] text-slate-600 dark:text-slate-400">
        <span>{lang || "text"}</span>
        <CopyButton text={code} />
      </div>
      <pre className="code-block-body bg-slate-100 dark:bg-slate-900 [padding:12px] [margin:0px] overflow-auto [max-height:600px]">
        {highlighted ? (
          <code
            dangerouslySetInnerHTML={{
              __html: highlighted,
            }}
          />
        ) : (
          <code className="font-mono text-[13px] text-slate-600 dark:text-slate-400">
            {code}
          </code>
        )}
      </pre>
    </div>
  )
}

/** Streaming cursor character */
const STREAMING_CURSOR = "\u258D" // ▍

export interface MarkdownContentProps {
  readonly content: string
  readonly isStreaming?: boolean
  readonly showCursor?: boolean
  readonly className?: string
  readonly style?: React.CSSProperties
}
function MarkdownContentImpl({
  content,
  isStreaming = false,
  showCursor = false,
  className,
  style,
}: MarkdownContentProps): ReactNode {
  const components = useMemo<Components>(
    () => ({
      code(props) {
        const { className: cls, children } = props
        const match = /language-(\w+)/.exec(cls || "")
        const isInline = !match && !String(children).includes("\n")
        if (isInline) {
          return (
            <code className="font-mono text-[13px] bg-slate-100 dark:bg-slate-900 [padding:1px_4px] rounded-[3px] text-slate-900 dark:text-slate-200">
              {children}
            </code>
          )
        }
        const lang = match?.[1] || "text"
        const code = String(children).replace(/\n$/, "")
        return <CodeBlock code={code} lang={lang} />
      },
      a({ href, children }) {
        return (
          <a
            href={href}
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-700 no-underline hover:underline dark:text-blue-400"
          >
            {children}
          </a>
        )
      },
      h1: (p) => (
        <h1
          {...p}
          className="font-[650] text-slate-900 dark:text-slate-200 text-[18px] [margin:16px_0_8px]"
        />
      ),
      h2: (p) => (
        <h2
          {...p}
          className="font-[650] text-slate-900 dark:text-slate-200 text-[16px] [margin:14px_0_8px]"
        />
      ),
      h3: (p) => (
        <h3
          {...p}
          className="font-[650] text-slate-900 dark:text-slate-200 text-[15px] [margin:12px_0_6px]"
        />
      ),
      h4: (p) => (
        <h4
          {...p}
          className="font-[650] text-slate-900 dark:text-slate-200 text-[14px] [margin:10px_0_6px]"
        />
      ),
      h5: (p) => (
        <h5
          {...p}
          className="font-[650] text-slate-900 dark:text-slate-200 text-[13px] [margin:10px_0_4px]"
        />
      ),
      h6: (p) => (
        <h6
          {...p}
          className="font-semibold text-slate-600 dark:text-slate-400 text-[13px] [margin:10px_0_4px]"
        />
      ),
      p: (p) => <p {...p} className="[margin:0_0_12px] leading-[1.55]" />,
      ul: (p) => (
        <ul
          {...p}
          className="font-sans [margin:0_0_12px] [padding-left:20px]"
        />
      ),
      ol: (p) => (
        <ol
          {...p}
          className="font-sans [margin:0_0_12px] [padding-left:20px]"
        />
      ),
      li: (p) => (
        <li
          {...p}
          className="[margin-bottom:4px] text-slate-900 dark:text-slate-200"
        />
      ),
      blockquote: (p) => (
        <blockquote
          {...p}
          className="border-l-[3px] border-l-slate-300 dark:border-l-slate-750 [padding-left:12px] [margin:8px_0_12px] text-slate-600 dark:text-slate-400"
        />
      ),
      table: (p) => (
        <table
          {...p}
          className="border border-slate-300 dark:border-slate-750 border-collapse w-full [margin:8px_0_12px] text-[13px]"
        />
      ),
      thead: (p) => <thead {...p} className="bg-white dark:bg-slate-850" />,
      th: (p) => (
        <th
          {...p}
          className="border border-slate-300 dark:border-slate-750 [padding:6px_8px] text-left font-semibold"
        />
      ),
      td: (p) => (
        <td
          {...p}
          className="border border-slate-300 dark:border-slate-750 [padding:6px_8px]"
        />
      ),
      hr: () => (
        <hr className="my-3 border-t border-slate-300 dark:border-slate-750" />
      ),
      strong: (p) => (
        <strong
          {...p}
          className="font-semibold text-slate-900 dark:text-slate-200"
        />
      ),
      em: (p) => <em {...p} className="italic" />,
    }),
    []
  )
  const displayContent = useMemo(() => {
    if (showCursor && isStreaming) {
      return content + STREAMING_CURSOR
    }
    return content
  }, [content, showCursor, isStreaming])
  return (
    <div
      className={`${
        className ?? "markdown-content"
      }  font-sans text-[14px] text-slate-900 dark:text-slate-200 leading-[1.55]`}
      style={{
        ...style,
      }}
    >
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {displayContent}
      </ReactMarkdown>
    </div>
  )
}
export const MarkdownContent = memo(MarkdownContentImpl)
