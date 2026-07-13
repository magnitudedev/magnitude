import type { CSSProperties, ReactNode } from "react"
import { ArrowLeft } from "lucide-react"

export interface ChatColumnPageProps {
  title: ReactNode
  backLabel?: string
  onBack: () => void
  children: ReactNode
  actions?: ReactNode
  bodyStyle?: CSSProperties
}

export function ChatColumnPage({
  title,
  backLabel = "Back to session",
  onBack,
  children,
  actions,
  bodyStyle,
}: ChatColumnPageProps): ReactNode {
  return (
    <>
      <div className="chat-title-bar">
        <button
          type="button"
          onClick={onBack}
          aria-label={backLabel}
          title={backLabel}
          className="hover-surface hover-fg"
          style={{
            width: 28,
            height: 28,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            background: "transparent",
            border: "none",
            borderRadius: 4,
            cursor: "pointer",
            flexShrink: 0,
            marginRight: 8,
          }}
        >
          <ArrowLeft size={16} />
        </button>
        <span className="chat-title-bar-title">{title}</span>
        {actions ? (
          <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8 }}>
            {actions}
          </div>
        ) : null}
      </div>
      <div
        style={{
          flex: 1,
          minHeight: 0,
          display: "flex",
          flexDirection: "column",
          ...bodyStyle,
        }}
      >
        {children}
      </div>
    </>
  )
}
