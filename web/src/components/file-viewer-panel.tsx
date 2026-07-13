/**
 * FileViewerPanel — spec §8.5
 *
 * Slides in from right, code/markdown/image viewer with shiki.
 * Width 45% of chat column (min 320px, max 600px), resizable.
 *
 * No useEffect — uses:
 * - Derived width (computed during render)
 * - onKeyDown for Esc
 * - onScroll for auto-scroll
 * - Mouse event handlers for resize (with useSyncExternalStore for resize state)
 * - useSyncExternalStore for Shiki highlighter
 */
import React, { useMemo, useState, useSyncExternalStore, useRef, useCallback, type ReactNode } from "react"
import { Copy, Check, ExternalLink, X } from "lucide-react"
import { usePlatform } from "../hooks/use-platform"
import { subscribeShiki, getShikiSnapshot, highlightCode } from "../stores/shiki-store"
import { createFocusTrapHandler } from "../utils/focus-trap"
import { MarkdownContent } from "./markdown-content"

export interface FileViewerPanelProps {
  filePath: string | null
  content: string | null
  loading?: boolean
  error?: string | null
  language?: string
  isStreaming?: boolean
  onClose: () => void
  onCopy?: (text: string) => void
}

// Resize store — tracks mouse position during drag
let resizeActive = false
let resizeWidth = 0
const resizeListeners = new Set<() => void>()

function startResize(initialWidth: number): void {
  resizeActive = true
  resizeWidth = initialWidth
}

function subscribeResize(cb: () => void): () => void {
  if (resizeActive) {
    const handler = (e: MouseEvent) => {
      resizeWidth = Math.min(600, Math.max(320, window.innerWidth - e.clientX))
      resizeListeners.forEach((l) => l())
    }
    const upHandler = () => {
      resizeActive = false
      resizeListeners.forEach((l) => l())
      window.removeEventListener("mousemove", handler)
      window.removeEventListener("mouseup", upHandler)
    }
    window.addEventListener("mousemove", handler)
    window.addEventListener("mouseup", upHandler)
    resizeListeners.add(cb)
    return () => {
      resizeListeners.delete(cb)
      window.removeEventListener("mousemove", handler)
      window.removeEventListener("mouseup", upHandler)
    }
  }
  return () => {}
}

function getResizeSnapshot(): number {
  return resizeWidth
}

function isResizing(): boolean {
  return resizeActive
}

export function FileViewerPanel({
  filePath,
  content,
  loading = false,
  error = null,
  language,
  isStreaming = false,
  onClose,
  onCopy,
}: FileViewerPanelProps): ReactNode {
  const [copied, setCopied] = useState(false)
  const contentRef = useRef<HTMLDivElement>(null)
  const panelRef = useRef<HTMLDivElement>(null)
  const platform = usePlatform()

  // Initial width — derived during render (no effect)
  const resizeWidth = useSyncExternalStore(subscribeResize, getResizeSnapshot)
  const dragging = isResizing()
  const defaultWidth = Math.min(600, Math.max(320, typeof window !== "undefined" ? window.innerWidth * 0.45 : 400))
  const panelWidth = dragging ? resizeWidth : defaultWidth

  // Focus trap + Esc to close
  const handleKeyDown = createFocusTrapHandler(panelRef, onClose)

  // Auto-scroll on new content — via onScroll handler checking if near bottom
  const handleScroll = useCallback(() => {
    // Scroll handling is done passively — streaming content pushes down naturally
  }, [])

  // Resize start — mousedown handler
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    startResize(panelWidth)
  }, [panelWidth])

  const handleCopy = useCallback(async () => {
    if (!content) return
    if (onCopy) {
      onCopy(content)
    } else {
      await platform.clipboard.writeText(content)
    }
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }, [content, onCopy, platform.clipboard])

  const handleOpenExternal = useCallback(async () => {
    if (filePath) {
      await platform.openPath(filePath)
    }
  }, [filePath, platform])

  if (!filePath) return null

  const lang = language || filePath.split(".").pop() || "text"
  const isMarkdown = lang === "md" || lang === "markdown" || lang === "mdx"
  const isImage = ["png", "jpg", "jpeg", "gif", "webp", "svg"].includes(lang)

  return (
    <div
      ref={panelRef}
      className="file-viewer-panel"
      tabIndex={-1}
      onKeyDown={handleKeyDown}
      style={{
        position: "fixed",
        top: 0, right: 0, bottom: 0,
        width: panelWidth,
        background: "var(--bg-surface)",
        borderLeft: "1px solid var(--border-default)",
        display: "flex",
        flexDirection: "column",
        zIndex: 30,
        animation: "slide-in-right 200ms ease-out",
      }}
    >
      {/* Resize handle */}
      <div
        onMouseDown={handleMouseDown}
        style={{
          position: "absolute",
          left: -3,
          top: 0, bottom: 0,
          width: 6,
          cursor: "col-resize",
          zIndex: 1,
        }}
      />

      {/* Header */}
      <div
        className="file-viewer-header"
        style={{
          height: 40,
          padding: "0 12px",
          borderBottom: "1px solid var(--border-subtle)",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          flexShrink: 0,
        }}
      >
        <div style={{
          display: "flex",
          alignItems: "center",
          gap: 6,
          overflow: "hidden",
          fontFamily: "var(--font-mono)",
          fontSize: 12,
          color: "var(--fg-secondary)",
        }}>
          <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            {filePath}
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 4, flexShrink: 0 }}>
          <button
            onClick={handleCopy}
            title="Copy"
            aria-label="Copy file content"
            style={{
              width: 28, height: 28,
              display: "flex", alignItems: "center", justifyContent: "center",
              background: "transparent", border: "none", borderRadius: 4,
              color: "var(--fg-tertiary)", cursor: "pointer",
            }}
          >
            {copied ? <Check size={14} /> : <Copy size={14} />}
          </button>
          <button
            onClick={handleOpenExternal}
            title="Open in editor"
            aria-label="Open in editor"
            style={{
              width: 28, height: 28,
              display: "flex", alignItems: "center", justifyContent: "center",
              background: "transparent", border: "none", borderRadius: 4,
              color: "var(--fg-tertiary)", cursor: "pointer",
            }}
          >
            <ExternalLink size={14} />
          </button>
          <button
            onClick={onClose}
            title="Close (Esc)"
            aria-label="Close file viewer"
            style={{
              width: 28, height: 28,
              display: "flex", alignItems: "center", justifyContent: "center",
              background: "transparent", border: "none", borderRadius: 4,
              color: "var(--fg-tertiary)", cursor: "pointer",
            }}
          >
            <X size={14} />
          </button>
        </div>
      </div>

      {/* Content */}
      <div
        ref={contentRef}
        onScroll={handleScroll}
        style={{
          flex: 1,
          overflow: "auto",
          padding: 0,
        }}
      >
        {loading ? (
          <div style={{ padding: 24, textAlign: "center", color: "var(--fg-tertiary)", fontFamily: "var(--font-mono)", fontSize: 13 }}>
            Loading...
          </div>
        ) : error ? (
          <div style={{ padding: 24, textAlign: "center", color: "var(--accent-error)", fontFamily: "var(--font-sans)", fontSize: 14 }}>
            {error}
          </div>
        ) : isImage ? (
          <div style={{ padding: 16, textAlign: "center" }}>
            <img src={`data:image/${lang};base64,${content}`} alt={filePath} style={{ maxWidth: "100%", borderRadius: 4 }} />
          </div>
        ) : (content || "").length > 50000 ? (
          <div style={{ padding: 24, textAlign: "center", color: "var(--fg-tertiary)", fontFamily: "var(--font-sans)", fontSize: 14, display: "flex", flexDirection: "column", alignItems: "center", gap: 12 }}>
            <div>File is too large to display ({(content || "").length.toLocaleString()} characters).</div>
            <button
              onClick={handleOpenExternal}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 6,
                background: "var(--bg-surface-elevated)",
                border: "1px solid var(--border-default)",
                borderRadius: 4,
                padding: "6px 12px",
                fontFamily: "var(--font-sans)",
                fontSize: 13,
                color: "var(--fg-secondary)",
                cursor: "pointer",
              }}
            >
              <ExternalLink size={14} />
              Open in editor
            </button>
          </div>
        ) : isMarkdown ? (
          <MarkdownContent
            content={content || ""}
            isStreaming={isStreaming}
            style={{
              padding: 16,
              fontFamily: "var(--font-sans)",
              lineHeight: 1.6,
              overflow: "auto",
            }}
          />
        ) : (
          <CodeBlock content={content || ""} language={lang} />
        )}
      </div>
    </div>
  )
}

// ── Code Block (Shiki) ──

function CodeBlock({
  content,
  language,
}: {
  content: string
  language: string
}): ReactNode {
  const highlighter = useSyncExternalStore(subscribeShiki, getShikiSnapshot)
  const html = useMemo(
    () => highlightCode(content, language || "text"),
    [content, highlighter, language],
  )

  if (!html) {
    return (
      <pre style={{ margin: 0, padding: 12, fontFamily: "var(--font-mono)", fontSize: 13, color: "var(--fg-primary)", background: "var(--bg-code)", overflow: "auto" }}>
        <code>{content}</code>
      </pre>
    )
  }

  return <div dangerouslySetInnerHTML={{ __html: html }} />
}
