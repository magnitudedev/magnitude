import { useRef, useState, type KeyboardEvent, type PointerEvent, type ReactNode } from "react"

import { cn } from "@/lib/utils"
import { clampResizableValue } from "@/lib/resizable"

export function ResizableEdge({
  side,
  value,
  minimum,
  maximum,
  onValueChange,
  onDraggingChange,
  label,
  className,
}: {
  readonly side: "left" | "right"
  readonly value: number
  readonly minimum: number
  readonly maximum: number
  readonly onValueChange: (value: number) => void
  readonly onDraggingChange?: (dragging: boolean) => void
  readonly label: string
  readonly className?: string
}): ReactNode {
  const drag = useRef<{ readonly pointerId: number; readonly startX: number; readonly startValue: number } | null>(null)
  const [dragging, setDragging] = useState(false)
  const updateDragging = (next: boolean) => {
    setDragging(next)
    onDraggingChange?.(next)
  }
  const stopDragging = () => {
    if (drag.current === null) return
    drag.current = null
    updateDragging(false)
  }
  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const step = event.shiftKey ? 48 : 16
    const physicalDirection = event.key === "ArrowLeft" ? -1 : event.key === "ArrowRight" ? 1 : 0
    if (physicalDirection !== 0) {
      event.preventDefault()
      const widthDirection = side === "right" ? physicalDirection : -physicalDirection
      onValueChange(clampResizableValue(value + widthDirection * step, minimum, maximum))
      return
    }
    if (event.key === "Home" || event.key === "End") {
      event.preventDefault()
      onValueChange(event.key === "Home" ? minimum : maximum)
    }
  }

  return (
    <div
      role="separator"
      aria-label={label}
      aria-orientation="vertical"
      aria-valuemin={minimum}
      aria-valuemax={maximum}
      aria-valuenow={clampResizableValue(value, minimum, maximum)}
      aria-valuetext={`${clampResizableValue(value, minimum, maximum)} pixels`}
      tabIndex={0}
      data-dragging={dragging || undefined}
      className={cn(
        "absolute inset-y-0 z-30 w-2 touch-none cursor-col-resize outline-none [-webkit-app-region:no-drag] [&:hover>span]:bg-blue-400 [&:focus-visible>span]:bg-blue-500 data-[dragging=true]:[&>span]:bg-blue-500 dark:[&:hover>span]:bg-blue-500 dark:[&:focus-visible>span]:bg-blue-400 dark:data-[dragging=true]:[&>span]:bg-blue-400",
        side === "left" ? "-left-1" : "-right-1",
        className,
      )}
      onKeyDown={handleKeyDown}
      onPointerDown={(event: PointerEvent<HTMLDivElement>) => {
        if (event.button !== 0) return
        event.preventDefault()
        drag.current = { pointerId: event.pointerId, startX: event.clientX, startValue: value }
        event.currentTarget.setPointerCapture(event.pointerId)
        updateDragging(true)
      }}
      onPointerMove={(event: PointerEvent<HTMLDivElement>) => {
        const current = drag.current
        if (current === null || current.pointerId !== event.pointerId) return
        const physicalDelta = event.clientX - current.startX
        const widthDelta = side === "right" ? physicalDelta : -physicalDelta
        onValueChange(clampResizableValue(current.startValue + widthDelta, minimum, maximum))
      }}
      onPointerUp={(event: PointerEvent<HTMLDivElement>) => {
        if (drag.current?.pointerId !== event.pointerId) return
        if (event.currentTarget.hasPointerCapture(event.pointerId)) {
          event.currentTarget.releasePointerCapture(event.pointerId)
        }
        stopDragging()
      }}
      onPointerCancel={stopDragging}
      onLostPointerCapture={stopDragging}
    >
      <span className="pointer-events-none absolute inset-y-0 left-1/2 w-px -translate-x-1/2 bg-transparent transition-colors" />
    </div>
  )
}
