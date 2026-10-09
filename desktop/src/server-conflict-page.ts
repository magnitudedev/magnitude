import type { ServerConflictApi } from "./server-conflict-preload"

/** The one screen shown while the server runs; the main process owns both decisions. */
const bridge = (window as unknown as { readonly __magnitudeServerConflict: ServerConflictApi }).__magnitudeServerConflict
const element = <T extends HTMLElement>(id: string) => document.getElementById(id) as T
const keep = element<HTMLButtonElement>("keep"), stop = element<HTMLButtonElement>("stop"), status = element<HTMLParagraphElement>("status")

// Linux keeps the server's data in its own account; macOS serves from the person's own data.
element<HTMLParagraphElement>("data").textContent = bridge.platform === "linux"
  ? "Stopping keeps the server's data. The app uses your own models, so models downloaded on the server won't be in it."
  : "The app uses the same models as the server."

element<HTMLAnchorElement>("address").addEventListener("click", event => {
  event.preventDefault()
  void bridge.keep()
})
keep.addEventListener("click", () => {
  keep.disabled = stop.disabled = true
  void bridge.keep()
})
stop.addEventListener("click", () => {
  keep.disabled = stop.disabled = true
  status.className = ""
  status.textContent = "Stopping the server…"
  void bridge.stop().then(result => {
    if (result.ok) return
    keep.disabled = stop.disabled = false
    status.className = "error"
    status.textContent = result.message
  })
})
