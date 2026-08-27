import { type ReactNode } from "react"
import { TextAttributes } from "@opentui/core"
import { wrapTextToWordLines } from "@magnitudedev/client-common"
import { useTheme } from "../../hooks/use-theme"

export type SetupStage = "choose" | "install" | "harness"

const SETUP_CONTENT_MAX_WIDTH = 110
const SETUP_WIDE_MIN_WIDTH = 105
const SETUP_STEP_LABELS = ["Choose model", "Install model", "Select harness"] as const
const SETUP_STEP_MARKER_COLUMNS = 2
const SETUP_STEP_CONNECTOR_COLUMNS = 6
const HORIZONTAL_SETUP_STEPPER_COLUMNS = SETUP_STEP_LABELS.reduce(
  (columns, label) => columns + SETUP_STEP_MARKER_COLUMNS + label.length,
  SETUP_STEP_CONNECTOR_COLUMNS * (SETUP_STEP_LABELS.length - 1),
)
const WIDE_SETUP_FRAME_ROWS = 26
const STACKED_SETUP_FRAME_ROWS = 44
const VERTICAL_STEPPER_ADDITIONAL_ROWS = 4
const SETUP_FOOTER_ROWS = 2

export const setupContentWidth = (viewportWidth: number): number =>
  Math.max(1, Math.min(SETUP_CONTENT_MAX_WIDTH, viewportWidth - 2))

export const setupBodyWidth = (viewportWidth: number): number =>
  Math.max(1, setupContentWidth(viewportWidth) - 4)

export const isWideSetupLayout = (viewportWidth: number): boolean =>
  setupContentWidth(viewportWidth) >= SETUP_WIDE_MIN_WIDTH

export const isHorizontalSetupStepper = (viewportWidth: number): boolean =>
  setupBodyWidth(viewportWidth) >= HORIZONTAL_SETUP_STEPPER_COLUMNS

export const setupFrameHeight = (viewportWidth: number, additionalRows = 0): number =>
  (isWideSetupLayout(viewportWidth)
    ? WIDE_SETUP_FRAME_ROWS
    : STACKED_SETUP_FRAME_ROWS
      + (isHorizontalSetupStepper(viewportWidth) ? 0 : VERTICAL_STEPPER_ADDITIONAL_ROWS))
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

  if (vertical) {
    return (
      <box style={{
        flexDirection: "column",
        height: 5,
        minHeight: 5,
        maxHeight: 5,
        flexShrink: 0,
        marginBottom: 1,
      }}>
        {SETUP_STEP_LABELS.map((label, position) => (
          <box key={label} style={{ flexDirection: "column", flexShrink: 0 }}>
            <text style={{ fg: position === activeIndex ? theme.accent : theme.text.body }}>
              {position <= activeIndex ? "●" : "○"} {label}
            </text>
            {position < SETUP_STEP_LABELS.length - 1 && (
              <text style={{ fg: position < activeIndex ? theme.text.body : theme.text.supporting }}>│</text>
            )}
          </box>
        ))}
      </box>
    )
  }

  return (
    <box style={{
      flexDirection: "row",
      height: 1,
      minHeight: 1,
      maxHeight: 1,
      flexShrink: 0,
      marginBottom: 1,
    }}>
      {SETUP_STEP_LABELS.map((label, position) => (
        <text key={label} style={{ fg: position === activeIndex ? theme.accent : theme.text.body }}>
          {position <= activeIndex ? "●" : "○"} {label}{position < SETUP_STEP_LABELS.length - 1 ? ` ${position < activeIndex ? "════" : "────"} ` : ""}
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
        <text
          style={{ fg: theme.text.body, height: 1, minHeight: 1, maxHeight: 1, flexShrink: 0 }}
          attributes={TextAttributes.BOLD}
        >
          MAGNITUDE SETUP
        </text>
        <SetupStepper stage={stage} vertical={!isHorizontalSetupStepper(width)} />
        <scrollbox
          focusable={false}
          scrollX={false}
          scrollbarOptions={{ visible: false }}
          verticalScrollbarOptions={{ visible: false }}
          style={{
            flexGrow: 1,
            minHeight: 0,
            rootOptions: {
              flexGrow: 1,
              minHeight: 0,
              backgroundColor: "transparent",
            },
            wrapperOptions: { border: false, backgroundColor: "transparent" },
            viewportOptions: { backgroundColor: "transparent" },
            contentOptions: { flexDirection: "column" },
          }}
        >
          {children}
          <box style={{ flexGrow: 1 }} />
        </scrollbox>
        {footer !== undefined && <box style={{ height: 1, minHeight: 1, flexShrink: 0 }} />}
        {footer !== undefined && (
          <box style={{
            height: SETUP_FOOTER_ROWS,
            minHeight: SETUP_FOOTER_ROWS,
            maxHeight: SETUP_FOOTER_ROWS,
            flexDirection: "column",
            flexShrink: 0,
            overflow: "hidden",
          }}>
            {footer}
          </box>
        )}
        {unexpectedErrorLines.map((line, index) => (
          <text key={`${index}:${line}`} style={{ fg: theme.status.failure }} wrapMode="none">
            {line}
          </text>
        ))}
      </box>
    </box>
  )
}
