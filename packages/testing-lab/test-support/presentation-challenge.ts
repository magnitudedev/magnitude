import type { Page } from "playwright"

/** Deliberately vary only presentation on a real app. Never mutate test IDs or semantic state. */
export const challengePresentation = (page: Page) => page.evaluate(() => {
  const rename = () => {
    for (const element of document.querySelectorAll<HTMLElement>("button, input, h1, h2")) {
      if (element.hasAttribute("aria-label")) element.setAttribute("aria-label", "Reworded accessible label")
      if (element instanceof HTMLInputElement) element.placeholder = "Different placeholder"
      if (element.tagName === "BUTTON") {
        element.style.backgroundColor = "rgb(184, 135, 230)"
        element.style.color = "rgb(10, 10, 10)"
        element.style.borderRadius = "0px"
      }
      const text = document.createTreeWalker(element, NodeFilter.SHOW_TEXT)
      for (let node = text.nextNode(); node; node = text.nextNode()) {
        if (node.textContent?.trim() && node.textContent !== "Different copy") node.textContent = "Different copy"
      }
    }
    for (const nav of document.querySelectorAll<HTMLElement>("nav")) nav.style.flexDirection = "column-reverse"
    for (const group of document.querySelectorAll<HTMLElement>('[role="group"]')) group.style.flexDirection = "row-reverse"
  }
  const observer = new MutationObserver(() => {
    observer.disconnect()
    rename()
    observer.observe(document.body, { childList: true, subtree: true, characterData: true })
  })
  rename()
  observer.observe(document.body, { childList: true, subtree: true, characterData: true })
})
