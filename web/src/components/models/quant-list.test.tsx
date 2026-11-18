import { renderToStaticMarkup } from "react-dom/server"
import { expect, it } from "vitest"
import { QuantList } from "./parts"
import type { QuantView } from "./view"

const quant = {
  key: "hf:bartowski/LFM3-1.7B-GGUF/LFM3-1.7B-BF16.gguf", name: "BF16", publisher: "bartowski", repository: "bartowski/LFM3-1.7B-GGUF",
  bits: 16, fits: true, measuring: false, speed: 98, sizeBytes: 3.47e9,
} as unknown as QuantView

it("chooses a size when its name is clicked, opening Hugging Face only from the icon beside it", () => {
  const html = renderToStaticMarkup(<QuantList quants={[quant]} selected={null} recommended={null} owned={new Set()} onSelect={() => {}} />)
  expect(html).toMatch(/<span class="truncate">bartowski BF16<\/span><a href="https:\/\/huggingface.co\/bartowski\/LFM3-1.7B-GGUF"[^>]*aria-label="Open bartowski\/LFM3-1.7B-GGUF on Hugging Face"/)
  expect(html).toContain('aria-label="16-bit bartowski BF16"')
  expect(html).not.toMatch(/<button\b(?:(?!<\/button>).)*<a\b/s)
})
