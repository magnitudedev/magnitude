import { Button as ButtonPrimitive } from "@base-ui/react/button"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

const buttonVariants = cva(
  "group/button inline-flex shrink-0 items-center justify-center rounded-md border border-transparent bg-clip-padding text-[13px] font-medium whitespace-nowrap transition-colors outline-none select-none focus-visible:border-blue-600 focus-visible:ring-1 focus-visible:ring-blue-600/30 active:not-aria-[haspopup]:translate-y-px disabled:pointer-events-none disabled:opacity-50 aria-invalid:border-red-600 aria-invalid:ring-1 aria-invalid:ring-red-600/20 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4 dark:focus-visible:border-blue-500 dark:focus-visible:ring-blue-500/30 dark:aria-invalid:border-red-400 dark:aria-invalid:ring-red-400/30",
  {
    variants: {
      variant: {
        default:
          "bg-blue-700 text-white hover:bg-blue-800 dark:bg-blue-500 dark:text-slate-925 dark:hover:bg-blue-400",
        outline:
          "border-slate-300 bg-white text-slate-700 hover:bg-slate-100 aria-expanded:bg-slate-100 dark:border-slate-700 dark:bg-slate-850 dark:text-slate-300 dark:hover:bg-slate-750 dark:aria-expanded:bg-slate-750",
        secondary:
          "bg-slate-100 text-slate-800 hover:bg-slate-150 aria-expanded:bg-slate-150 dark:bg-slate-800 dark:text-slate-200 dark:hover:bg-slate-750 dark:aria-expanded:bg-slate-750",
        ghost:
          "text-slate-600 hover:bg-slate-150 hover:text-slate-900 aria-expanded:bg-slate-150 dark:text-slate-400 dark:hover:bg-slate-750 dark:hover:text-slate-200 dark:aria-expanded:bg-slate-750",
        destructive:
          "bg-red-600 text-white hover:bg-red-700 focus-visible:border-red-600 focus-visible:ring-red-600/30 dark:bg-red-500 dark:text-slate-925 dark:hover:bg-red-400 dark:focus-visible:border-red-400 dark:focus-visible:ring-red-400/30",
        link: "text-slate-850 underline-offset-4 hover:underline dark:text-slate-200",
        unstyled: "",
      },
      size: {
        default:
          "h-8 gap-1.5 px-2.5 has-data-[icon=inline-end]:pr-2 has-data-[icon=inline-start]:pl-2",
        xs: "h-6 gap-1 rounded-md px-2 text-xs has-data-[icon=inline-end]:pr-1.5 has-data-[icon=inline-start]:pl-1.5 [&_svg:not([class*='size-'])]:size-3",
        sm: "h-7 gap-1 rounded-md px-2.5 has-data-[icon=inline-end]:pr-1.5 has-data-[icon=inline-start]:pl-1.5 [&_svg:not([class*='size-'])]:size-3.5",
        lg: "h-9 gap-1.5 px-2.5 has-data-[icon=inline-end]:pr-2 has-data-[icon=inline-start]:pl-2",
        icon: "size-8",
        "icon-xs": "size-6 rounded-md [&_svg:not([class*='size-'])]:size-3",
        "icon-sm": "size-7 rounded-md",
        "icon-lg": "size-9",
        unstyled: "",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

function Button({
  className,
  variant = "default",
  size = "default",
  ...props
}: ButtonPrimitive.Props & VariantProps<typeof buttonVariants>) {
  return (
    <ButtonPrimitive
      data-slot="button"
      className={
        variant === "unstyled"
          ? cn(
              "disabled:pointer-events-none disabled:opacity-50",
              className
            )
          : cn(buttonVariants({ variant, size, className }))
      }
      {...props}
    />
  )
}

export { Button, buttonVariants }
