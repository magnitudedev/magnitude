import { Checkbox as CheckboxPrimitive } from "@base-ui/react/checkbox"

import { cn } from "@/lib/utils"
import { CheckIcon, MinusIcon } from "@phosphor-icons/react"

function Checkbox({ className, indeterminate, ...props }: CheckboxPrimitive.Root.Props) {
  return (
    <CheckboxPrimitive.Root
      data-slot="checkbox"
      className={cn(
        "peer relative flex size-4 shrink-0 items-center justify-center rounded-[4px] border border-slate-400 bg-white text-white outline-none transition-colors after:absolute after:-inset-x-3 after:-inset-y-2 focus-visible:border-blue-600 focus-visible:ring-1 focus-visible:ring-blue-600/30 disabled:cursor-not-allowed disabled:opacity-50 data-checked:border-blue-700 data-checked:bg-blue-700 dark:border-slate-600 dark:bg-slate-850 dark:focus-visible:border-blue-500 dark:focus-visible:ring-blue-500/30 dark:data-checked:border-blue-500 dark:data-checked:bg-blue-500 dark:data-checked:text-slate-925",
        className
      )}
      indeterminate={indeterminate}
      {...props}
    >
      <CheckboxPrimitive.Indicator
        data-slot="checkbox-indicator"
        className="grid place-content-center text-current transition-none [&>svg]:size-3.5"
      >
        {indeterminate ? <MinusIcon /> : <CheckIcon />}
      </CheckboxPrimitive.Indicator>
    </CheckboxPrimitive.Root>
  )
}

export { Checkbox }
