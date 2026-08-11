import { TextAttributes } from "@opentui/core"
import { Option } from "effect"
import { useState, type ReactNode } from "react"
import stringWidth from "string-width"
import type {
  NotificationAction,
  NotificationState,
} from "@magnitudedev/client-common"
import type { ChatTheme } from "../../types/theme-system"
import { Button } from "../../components/button"

export const notificationAreaLabel = (
  notificationState: NotificationState,
  compact = false,
): string => {
  const message = compact
    ? Option.getOrElse(
      notificationState.compactMessage,
      () => notificationState.message,
    )
    : notificationState.message
  return notificationState.priority === "warning"
    || notificationState.priority === "error"
    ? `! ${message}`
    : message
}

export const notificationAreaWidth = (
  notificationState: NotificationState | null,
  compact = false,
): number => notificationState === null
  ? 0
  : stringWidth(notificationAreaLabel(notificationState, compact))

const notificationColor = (
  notificationState: NotificationState,
  theme: ChatTheme,
): string => {
  switch (notificationState.priority) {
    case "activity": return theme.primary
    case "notice": return theme.foreground
    case "warning": return theme.warning
    case "error": return theme.error
  }
}

export function NotificationArea({
  notificationState,
  theme,
  onAction,
  compact = false,
}: {
  readonly notificationState: NotificationState
  readonly theme: ChatTheme
  readonly onAction: (action: NotificationAction) => void
  readonly compact?: boolean
}): ReactNode {
  const [hovered, setHovered] = useState(false)
  const label = notificationAreaLabel(notificationState, compact)
  const text = (
    <text style={{ fg: notificationColor(notificationState, theme) }}>
      <span attributes={hovered ? TextAttributes.UNDERLINE : TextAttributes.NONE}>
        {label}
      </span>
    </text>
  )
  return Option.match(notificationState.action, {
    onNone: () => text,
    onSome: (action) => (
      <Button
        onClick={() => onAction(action)}
        onMouseOver={() => setHovered(true)}
        onMouseOut={() => setHovered(false)}
      >
        {text}
      </Button>
    ),
  })
}
