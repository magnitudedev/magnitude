/**
 * Opens a page in the system browser. The desktop window hands every new-window request for an
 * https page to the operating system's browser; a browser opens it in a new tab.
 */
export const openExternalUrl = (url: string): void => {
  window.open(url, "_blank", "noopener,noreferrer")
}
