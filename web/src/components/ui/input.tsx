import * as React from "react"
import { Input as InputPrimitive } from "@base-ui/react/input"

import { cn } from "@/lib/utils"

function Input({ className, type, ...props }: React.ComponentProps<"input">) {
  return (
    <InputPrimitive
      type={type}
      data-slot="input"
      className={cn(
        "h-8 w-full min-w-0 rounded-md border border-slate-300 bg-white px-2.5 py-1 font-sans text-[13px] text-slate-900 outline-none transition-colors file:inline-flex file:h-6 file:border-0 file:bg-transparent file:text-[13px] file:font-medium file:text-slate-900 placeholder:text-slate-500 focus-visible:border-blue-500 focus-visible:ring-2 focus-visible:ring-blue-500/20 disabled:pointer-events-none disabled:cursor-not-allowed disabled:bg-slate-100 disabled:opacity-50 aria-invalid:border-red-600 aria-invalid:ring-2 aria-invalid:ring-red-600/20 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100 dark:file:text-slate-100 dark:placeholder:text-slate-500 dark:focus-visible:border-blue-400 dark:focus-visible:ring-blue-400/20 dark:disabled:bg-slate-800 dark:aria-invalid:border-red-400 dark:aria-invalid:ring-red-400/20",
        className
      )}
      {...props}
    />
  )
}

export { Input }
