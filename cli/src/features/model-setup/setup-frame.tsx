import { type ReactNode } from "react"
import { TextAttributes } from "@opentui/core"
import { wrapTextToWordLines } from "@magnitudedev/client-common"
import { useTheme } from "../../hooks/use-theme"

export type SetupStage = "choose" | "install" | "harness"

const SETUP_CONTENT_MAX_WIDTH = 110
const SETUP_WIDE_MIN_WIDTH = 105
const WIDE_SETUP_FRAME_ROWS = 26
const NARROW_SETUP_FRAME_ROWS = 42

export const setupContentWidth = (viewportWidth: number): number =>
  Math.max(1, Math.min(SETUP_CONTENT_MAX_WIDTH, viewportWidth - 2))

export const setupBodyWidth = (viewportWidth: number): number =>
  Math.max(1, setupContentWidth(viewportWidth) - 4)

export const isWideSetupLayout = (viewportWidth: number): boolean =>
  setupContentWidth(viewportWidth) >= SETUP_WIDE_MIN_WIDTH

export const setupFrameHeight = (viewportWidth: number, additionalRows = 0): number =>
  (isWideSetupLayout(viewportWidth) ? WIDE_SETUP_FRAME_ROWS : NARROW_SETUP_FRAME_ROWS)
    + additionalRows

export function SetupStepper({
  stage,
  vertical,
}: {
  readonly stage: SetupStage
  readonly vertical: boolean
}): ReactNode {
  const theme = useTheme()
  const activeIndex = stage === "choose" ? 0 : stage === "install" ? 1 : 2
  const labels = ["Choose model", "Install model", "Select harness"]

  if (vertical) {
    return (
      <box style={{ flexDirection: "column", marginBottom: 1 }}>
        {labels.map((label, position) => (
          <box key={label} style={{ flexDirection: "column" }}>
            <text style={{ fg: position === activeIndex ? theme.accent : theme.text.body }}>
              {position <= activeIndex ? "●" : "○"} {label}
            </text>
            {position < labels.length - 1 && (
              <text style={{ fg: position < activeIndex ? theme.text.body : theme.text.supporting }}>│</text>
            )}
          </box>
        ))}
      </box>
    )
  }

  return (
    <box style={{ flexDirection: "row", marginBottom: 1 }}>
      {labels.map((label, position) => (
        <text key={label} style={{ fg: position === activeIndex ? theme.accent : theme.text.body }}>
          {position <= activeIndex ? "●" : "○"} {label}{position < labels.length - 1 ? ` ${position < activeIndex ? "════" : "────"} ` : ""}
        </text>
      ))}
    </box>
  )
}

export function SetupFrame({
  width,
  stage,
  children,
  footer,
  unexpectedError,
  additionalRows = 0,
}: {
  readonly width: number
  readonly stage: SetupStage
  readonly children: ReactNode
  readonly footer?: ReactNode
  readonly unexpectedError?: string | null
  readonly additionalRows?: number
}): ReactNode {
  const theme = useTheme()
  const unexpectedErrorLines = unexpectedError === undefined || unexpectedError === null
    ? []
    : wrapTextToWordLines(unexpectedError, setupBodyWidth(width))
  return (
    <box style={{ width: "100%", flexGrow: 1, alignItems: "center", justifyContent: "center" }}>
      <box style={{
        width: setupContentWidth(width),
        height: setupFrameHeight(width, additionalRows + unexpectedErrorLines.length),
        maxHeight: "100%",
        flexDirection: "column",
        flexShrink: 1,
        paddingLeft: 2,
        paddingRight: 2,
      }}>
        <text style={{ fg: theme.text.body }} attributes={TextAttributes.BOLD}>MAGNITUDE SETUP</text>
        <SetupStepper stage={stage} vertical={!isWideSetupLayout(width)} />
        {children}
        <box style={{ flexGrow: 1 }} />
        {footer !== undefined && <box style={{ height: 1, minHeight: 1, flexShrink: 0 }} />}
        {footer}
        {unexpectedErrorLines.map((line, index) => (
          <text key={`${index}:${line}`} style={{ fg: theme.status.failure }} wrapMode="none">
            {line}
          </text>
        ))}
      </box>
    </box>
  )
}
