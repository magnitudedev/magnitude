import type { ReactNode } from "react"
import lightMarkUrl from "../../../assets/brand/icon-light.svg"
import darkMarkUrl from "../../../assets/brand/icon-dark.svg"

export function MagnitudeMark({
  className,
}: {
  readonly className?: string
}): ReactNode {
  return (
    <span aria-hidden="true" className={className}>
      <img src={lightMarkUrl} alt="" className="h-full w-full object-contain dark:hidden" />
      <img src={darkMarkUrl} alt="" className="hidden h-full w-full object-contain dark:block" />
    </span>
  )
}
