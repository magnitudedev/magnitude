import type { ReactNode } from "react"
import magnitudeMarkUrl from "../../../assets/brand/magnitude-mark.svg"

export function MagnitudeMark({
  className,
}: {
  readonly className?: string
}): ReactNode {
  return (
    <img
      src={magnitudeMarkUrl}
      alt=""
      aria-hidden="true"
      className={className}
    />
  )
}
