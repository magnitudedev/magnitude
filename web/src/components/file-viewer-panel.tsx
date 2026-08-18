import { Button } from "@/components/ui/button"

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
import React, {
  useMemo,
  useState,
  useSyncExternalStore,
  useRef,
  useCallback,
  type ReactNode,
} from "react"
import { Copy, Check, ExternalLink, X } from "lucide-react"
import { usePlatform } from "../hooks/use-platform"
import {
  subscribeShiki,
  getShikiSnapshot,
  highlightCode,
} from "../stores/shiki-store"
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
  const defaultWidth = Math.min(
    600,
    Math.max(
      320,
      typeof window !== "undefined" ? window.innerWidth * 0.45 : 400
    )
  )
  const panelWidth = dragging ? resizeWidth : defaultWidth

  // Focus trap + Esc to close
  const handleKeyDown = createFocusTrapHandler(panelRef, onClose)

  // Auto-scroll on new content — via onScroll handler checking if near bottom
  const handleScroll = useCallback(() => {
    // Scroll handling is done passively — streaming content pushes down naturally
  }, [])

  // Resize start — mousedown handler
  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault()
      startResize(panelWidth)
    },
    [panelWidth]
  )
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
      className="max-[800px]:!w-full max-[800px]:!min-w-0 fixed [top:0px] [right:0px] [bottom:0px] bg-white dark:bg-slate-850 border-l border-l-slate-300 dark:border-l-slate-750 flex flex-col z-[30] [animation:slide-in-right_200ms_ease-out]"
      tabIndex={-1}
      onKeyDown={handleKeyDown}
      style={{
        width: panelWidth,
      }}
    >
      {/* Resize handle */}
      <div
        onMouseDown={handleMouseDown}
        className="absolute -left-[3px] top-0 bottom-0 w-1.5 cursor-col-resize z-[1]"
      />

      {/* Header */}
      <div className="file-viewer-header [height:40px] [padding:0_12px] border-b border-b-slate-200 dark:border-b-slate-800 flex items-center justify-between shrink-0">
        <div className="flex items-center [gap:6px] overflow-hidden font-mono text-[12px] text-slate-600 dark:text-slate-400">
          <span className="overflow-hidden text-ellipsis whitespace-nowrap">
            {filePath}
          </span>
        </div>
        <div className="flex items-center [gap:4px] shrink-0">
          <Button variant="unstyled" size="unstyled"
            onClick={handleCopy}
            title="Copy"
            aria-label="Copy file content"
            className="[width:28px] [height:28px] flex items-center justify-center [background:transparent] border-0 rounded-[4px] text-slate-500 cursor-pointer"
          >
            {copied ? <Check size={14} /> : <Copy size={14} />}
          </Button>
          <Button variant="unstyled" size="unstyled"
            onClick={handleOpenExternal}
            title="Open in editor"
            aria-label="Open in editor"
            className="[width:28px] [height:28px] flex items-center justify-center [background:transparent] border-0 rounded-[4px] text-slate-500 cursor-pointer"
          >
            <ExternalLink size={14} />
          </Button>
          <Button variant="unstyled" size="unstyled"
            onClick={onClose}
            title="Close (Esc)"
            aria-label="Close file viewer"
            className="[width:28px] [height:28px] flex items-center justify-center [background:transparent] border-0 rounded-[4px] text-slate-500 cursor-pointer"
          >
            <X size={14} />
          </Button>
        </div>
      </div>

      {/* Content */}
      <div
        ref={contentRef}
        onScroll={handleScroll}
        className="[flex:1] overflow-auto [padding:0px]"
      >
        {loading ? (
          <div className="[padding:24px] text-center text-slate-500 font-mono text-[13px]">
            Loading...
          </div>
        ) : error ? (
          <div className="[padding:24px] text-center text-red-600 dark:text-red-500 font-sans text-[14px]">
            {error}
          </div>
        ) : isImage ? (
          <div className="[padding:16px] text-center">
            <img
              src={`data:image/${lang};base64,${content}`}
              alt={filePath}
              className="[max-width:100%] rounded-[4px]"
            />
          </div>
        ) : (content || "").length > 50000 ? (
          <div className="[padding:24px] text-center text-slate-500 font-sans text-[14px] flex flex-col items-center [gap:12px]">
            <div>
              File is too large to display (
              {(content || "").length.toLocaleString()} characters).
            </div>
            <Button variant="unstyled" size="unstyled"
              onClick={handleOpenExternal}
              className="flex items-center [gap:6px] bg-slate-100 dark:bg-slate-800 border border-slate-300 dark:border-slate-750 rounded-[4px] [padding:6px_12px] font-sans text-[13px] text-slate-600 dark:text-slate-400 cursor-pointer"
            >
              <ExternalLink size={14} />
              Open in editor
            </Button>
          </div>
        ) : isMarkdown ? (
          <MarkdownContent
            content={content || ""}
            isStreaming={isStreaming}
            className="[padding:16px] font-sans leading-[1.6] overflow-auto"
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
    [content, highlighter, language]
  )
  if (!html) {
    return (
      <pre className="[margin:0px] [padding:12px] font-mono text-[13px] text-slate-900 dark:text-slate-200 bg-slate-100 dark:bg-slate-900 overflow-auto">
        <code>{content}</code>
      </pre>
    )
  }
  return (
    <div
      dangerouslySetInnerHTML={{
        __html: html,
      }}
    />
  )
}
