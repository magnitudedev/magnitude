/**
 * MarkdownContent — spec §14.1
 *
 * react-markdown + remark-gfm wrapper with shiki code highlighting.
 * Overrides for code blocks, links, headings, blockquotes, lists, tables.
 * Streaming cursor support.
 *
 * Uses useSyncExternalStore for the shared Shiki highlighter — no useEffect.
 */
import { memo, useSyncExternalStore, useMemo, useState, type ReactNode } from "react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import type { Components } from "react-markdown"
import { Copy, Check } from "lucide-react"
import { subscribeShiki, getShikiSnapshot, highlightCode } from "../stores/shiki-store"

/** Highlight a code block — returns highlighted HTML or null if not yet loaded */
function useCodeHighlight(code: string, lang: string): string | null {
  // Subscribe to shiki store — re-renders when highlighter loads
  const highlighter = useSyncExternalStore(subscribeShiki, getShikiSnapshot)
  return useMemo(
    () => highlightCode(code, lang),
    [code, highlighter, lang],
  )
}

/** Copy button with feedback state */
function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)
  return (
    <button
      className="code-block-copy"
      aria-label="Copy code"
      onClick={(e) => {
        e.stopPropagation()
        navigator.clipboard?.writeText(text)
        setCopied(true)
        setTimeout(() => setCopied(false), 2000)
      }}
      style={{
        background: "transparent",
        border: "none",
        cursor: "pointer",
        color: copied ? "var(--accent-success)" : "var(--fg-tertiary)",
        display: "flex",
        alignItems: "center",
        padding: "2px",
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
    <div className="code-block" style={{ margin: "8px 0 12px", borderRadius: "4px", overflow: "hidden" }}>
      <div
        className="code-block-header"
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          background: "var(--bg-surface)",
          borderBottom: "1px solid var(--border-default)",
          padding: "4px 10px",
          fontFamily: "var(--font-sans)",
          fontSize: "11px",
          color: "var(--fg-secondary)",
        }}
      >
        <span>{lang || "text"}</span>
        <CopyButton text={code} />
      </div>
      <pre
        className="code-block-body"
        style={{
          background: "var(--bg-code)",
          padding: "12px",
          margin: 0,
          overflow: "auto",
          maxHeight: "600px",
        }}
      >
        {highlighted ? (
          <code dangerouslySetInnerHTML={{ __html: highlighted }} />
        ) : (
          <code style={{ fontFamily: "var(--font-mono)", fontSize: "13px", color: "var(--fg-secondary)" }}>
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
  const components = useMemo<Components>(() => ({
    code(props) {
      const { className: cls, children } = props
      const match = /language-(\w+)/.exec(cls || "")
      const isInline = !match && !String(children).includes("\n")
      if (isInline) {
        return (
          <code
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: "13px",
              background: "var(--bg-code)",
              padding: "1px 4px",
              borderRadius: "3px",
              color: "var(--fg-primary)",
            }}
          >
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
          className="md-link"
        >
          {children}
        </a>
      )
    },

    h1: (p) => <h1 style={{ fontFamily: "var(--font-sans)", fontWeight: 650, color: "var(--fg-primary)", fontSize: "18px", margin: "16px 0 8px" }} {...p} />,
    h2: (p) => <h2 style={{ fontFamily: "var(--font-sans)", fontWeight: 650, color: "var(--fg-primary)", fontSize: "16px", margin: "14px 0 8px" }} {...p} />,
    h3: (p) => <h3 style={{ fontFamily: "var(--font-sans)", fontWeight: 650, color: "var(--fg-primary)", fontSize: "15px", margin: "12px 0 6px" }} {...p} />,
    h4: (p) => <h4 style={{ fontFamily: "var(--font-sans)", fontWeight: 650, color: "var(--fg-primary)", fontSize: "14px", margin: "10px 0 6px" }} {...p} />,
    h5: (p) => <h5 style={{ fontFamily: "var(--font-sans)", fontWeight: 650, color: "var(--fg-primary)", fontSize: "13px", margin: "10px 0 4px" }} {...p} />,
    h6: (p) => <h6 style={{ fontFamily: "var(--font-sans)", fontWeight: 600, color: "var(--fg-secondary)", fontSize: "13px", margin: "10px 0 4px" }} {...p} />,

    p: (p) => <p style={{ margin: "0 0 12px", lineHeight: 1.55 }} {...p} />,

    ul: (p) => <ul style={{ fontFamily: "var(--font-sans)", margin: "0 0 12px", paddingLeft: "20px" }} {...p} />,
    ol: (p) => <ol style={{ fontFamily: "var(--font-sans)", margin: "0 0 12px", paddingLeft: "20px" }} {...p} />,
    li: (p) => <li style={{ marginBottom: "4px", color: "var(--fg-primary)" }} {...p} />,

    blockquote: (p) => (
      <blockquote
        style={{
          borderLeft: "3px solid var(--border-default)",
          paddingLeft: "12px",
          margin: "8px 0 12px",
          color: "var(--fg-secondary)",
        }}
        {...p}
      />
    ),

    table: (p) => (
      <table
        style={{
          border: "1px solid var(--border-default)",
          borderCollapse: "collapse",
          width: "100%",
          margin: "8px 0 12px",
          fontSize: "13px",
        }}
        {...p}
      />
    ),
    thead: (p) => <thead style={{ background: "var(--bg-surface)" }} {...p} />,
    th: (p) => <th style={{ border: "1px solid var(--border-default)", padding: "6px 8px", textAlign: "left", fontWeight: 600 }} {...p} />,
    td: (p) => <td style={{ border: "1px solid var(--border-default)", padding: "6px 8px" }} {...p} />,

    hr: () => <hr style={{ border: "none", borderTop: "1px solid var(--border-default)", margin: "12px 0" }} />,

    strong: (p) => <strong style={{ fontWeight: 600, color: "var(--fg-primary)" }} {...p} />,
    em: (p) => <em style={{ fontStyle: "italic" }} {...p} />,
  }), [])

  const displayContent = useMemo(() => {
    if (showCursor && isStreaming) {
      return content + STREAMING_CURSOR
    }
    return content
  }, [content, showCursor, isStreaming])

  return (
    <div
      className={className ?? "markdown-content"}
      style={{
        fontFamily: "var(--font-sans)",
        fontSize: "14px",
        color: "var(--fg-primary)",
        lineHeight: 1.55,
        ...style,
      }}
    >
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={components}
      >
        {displayContent}
      </ReactMarkdown>
    </div>
  )
}

export const MarkdownContent = memo(MarkdownContentImpl)
