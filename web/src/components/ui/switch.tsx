import { Switch as SwitchPrimitive } from "@base-ui/react/switch"

import { cn } from "@/lib/utils"

function Switch({
  className,
  size = "default",
  ...props
}: SwitchPrimitive.Root.Props & {
  size?: "sm" | "default"
}) {
  return (
    <SwitchPrimitive.Root
      data-slot="switch"
      data-size={size}
      className={cn(
        "peer group/switch relative inline-flex shrink-0 items-center rounded-full border border-transparent transition-all outline-none group-has-[:focus-visible]/field-label:ring-0 after:absolute after:-inset-x-3 after:-inset-y-2 focus-visible:border-blue-600 focus-visible:ring-1 focus-visible:ring-blue-600/50 aria-invalid:border-red-500 aria-invalid:ring-1 aria-invalid:ring-red-500/20 data-[size=default]:h-[18.4px] data-[size=default]:w-[32px] data-[size=sm]:h-[14px] data-[size=sm]:w-[24px] data-checked:bg-blue-600 data-unchecked:bg-slate-200 data-disabled:cursor-not-allowed data-disabled:opacity-50 dark:focus-visible:border-blue-500 dark:focus-visible:ring-blue-500/50 dark:aria-invalid:border-red-800 dark:aria-invalid:ring-red-800/40 dark:data-checked:bg-blue-500 dark:data-unchecked:bg-slate-800",
        className
      )}
      {...props}
    >
      <SwitchPrimitive.Thumb
        data-slot="switch-thumb"
        className="pointer-events-none block rounded-full bg-white ring-0 transition-transform group-data-[size=default]/switch:size-4 group-data-[size=sm]/switch:size-3 group-data-[size=default]/switch:data-checked:translate-x-[calc(100%-2px)] group-data-[size=sm]/switch:data-checked:translate-x-[calc(100%-2px)] group-data-[size=default]/switch:data-unchecked:translate-x-0 group-data-[size=sm]/switch:data-unchecked:translate-x-0 dark:bg-slate-50"
      />
    </SwitchPrimitive.Root>
  )
}

export { Switch }
