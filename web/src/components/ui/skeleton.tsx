import type { ComponentProps } from "react"
import { cn } from "@/lib/utils"

// Decorative by itself; the enclosing loading region provides one accessible label.
export function Skeleton({ className, ...props }: ComponentProps<"span">) {
  return <span {...props} aria-hidden="true" data-slot="skeleton" className={cn("block rounded-md bg-slate-200 motion-safe:animate-pulse dark:bg-slate-750", className)} />
}
