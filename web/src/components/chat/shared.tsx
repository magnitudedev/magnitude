import type { ReactNode } from "react"
import { Result } from "@effect-atom/atom-react"
import { Cause, Option } from "effect"
import { Button } from "../ui/button"
export const none = Option.none
export const value = <A,>(option: Option.Option<A>) =>
  Option.getOrUndefined(option)
export const number = (n: number) =>
  n.toLocaleString(undefined, { maximumFractionDigits: 1 })
export const seconds = (ms: number) =>
  `${(ms / 1000).toFixed(ms < 1000 ? 2 : 1)}s`
export const titleCase = (text: string) =>
  text === "none" || text === "off"
    ? "Off"
    : text[0]?.toUpperCase() + text.slice(1)
const unspecified = "This action could not be completed. Please try again."
export const errorText = (result: Result.Result<unknown, unknown>) =>
  Result.isFailure(result)
    ? Option.match(Cause.failureOption(result.cause), {
        onNone: () => unspecified,
        onSome: (error) =>
          error && typeof error === "object" && "message" in error
            ? String(error.message)
            : unspecified,
      })
    : null
export function IconButton({
  label,
  children,
  onClick,
  pressed,
  disabled,
}: {
  label: string
  children: ReactNode
  onClick: () => void
  pressed?: boolean
  disabled?: boolean
}) {
  return (
    <Button
      type="button"
      variant="ghost"
      size="icon"
      aria-label={label}
      title={label}
      aria-pressed={pressed}
      disabled={disabled}
      onClick={onClick}
    >
      {children}
    </Button>
  )
}
