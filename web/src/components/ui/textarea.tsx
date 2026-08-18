import * as React from "react"

import { cn } from "@/lib/utils"

function Textarea({ className, ...props }: React.ComponentProps<"textarea">) {
  return (
    <textarea
      data-slot="textarea"
      className={cn(
        "field-sizing-content flex min-h-16 w-full rounded-md border border-slate-300 bg-white px-2.5 py-2 font-sans text-[13px] text-slate-900 outline-none transition-colors placeholder:text-slate-500 focus-visible:border-blue-500 focus-visible:ring-2 focus-visible:ring-blue-500/20 disabled:cursor-not-allowed disabled:bg-slate-100 disabled:opacity-50 aria-invalid:border-red-600 aria-invalid:ring-2 aria-invalid:ring-red-600/20 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100 dark:placeholder:text-slate-500 dark:focus-visible:border-blue-400 dark:focus-visible:ring-blue-400/20 dark:disabled:bg-slate-800 dark:aria-invalid:border-red-400 dark:aria-invalid:ring-red-400/20",
        className
      )}
      {...props}
    />
  )
}

export { Textarea }
