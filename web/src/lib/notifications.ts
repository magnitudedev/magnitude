import { toast } from "@/components/ui/toast"

export type NotificationKind = "success" | "error" | "info"

export function notify(kind: NotificationKind, message: string): string {
  return toast.add({
    title: message,
    type: kind,
    priority: kind === "error" ? "high" : "low",
    timeout: 5_000,
  })
}
