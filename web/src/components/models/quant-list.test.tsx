import { renderToStaticMarkup } from "react-dom/server"
import { describe, expect, it } from "vitest"
import { ExpandingModel, type OfferedModel } from "./expanding-model"
import { PlacementBar, QuantLadder, QuantList, visibleLadderRungs } from "./parts"
import type { QuantView } from "./view"

const quant = {
  key: "hf:bartowski/LFM3-1.7B-GGUF/LFM3-1.7B-BF16.gguf", name: "BF16", publisher: "bartowski", repository: "bartowski/LFM3-1.7B-GGUF",
  bits: 16, fits: true, measuring: false, speed: 98, sizeBytes: 3.47e9,
} as unknown as QuantView

const makeLadder = (...bits: number[]): readonly QuantView[] =>
  bits.map((b, index) => ({
    key: `m:gguf:r${index}`, name: `Q${b}`, publisher: "pub", repository: null,
    bits: b, fits: true, measuring: false, speed: 90 - index * 15, sizeBytes: (index + 1) * 20e9,
    missingDiskBytes: 0, servedContext: null, quality: null,
  }) as unknown as QuantView)

it("chooses a size when its name is clicked, opening Hugging Face only from the icon beside it", () => {
  const html = renderToStaticMarkup(<QuantList quants={[quant]} selected={null} recommended={null} owned={new Set()} onSelect={() => {}} />)
  expect(html).toMatch(/<span class="truncate">bartowski BF16<\/span><a href="https:\/\/huggingface.co\/bartowski\/LFM3-1.7B-GGUF"[^>]*aria-label="Open bartowski\/LFM3-1.7B-GGUF on Hugging Face"/)
  expect(html).toContain('aria-label="16-bit bartowski BF16"')
  expect(html).not.toMatch(/<button\b(?:(?!<\/button>).)*<a\b/s)
})

describe("visibleLadderRungs", () => {
  const six = makeLadder(2, 3, 3.5, 4, 4.5, 6)
  const bits = (list: readonly QuantView[]) => list.map(item => item.bits)

  it("keeps every rung when a ladder already has four or fewer", () => {
    const four = makeLadder(2, 3, 4, 6)
    expect(bits(visibleLadderRungs(four, four[1]!.key, new Set()))).toEqual([2, 3, 4, 6])
  })

  it("windows a six-rung ladder to four around the recommended rung (up to two faster, one smarter)", () => {
    expect(bits(visibleLadderRungs(six, six[0]!.key, new Set()))).toEqual([2, 3, 3.5, 4])
    expect(bits(visibleLadderRungs(six, six[1]!.key, new Set()))).toEqual([2, 3, 3.5, 4])
    expect(bits(visibleLadderRungs(six, six[2]!.key, new Set()))).toEqual([2, 3, 3.5, 4])
    expect(bits(visibleLadderRungs(six, six[3]!.key, new Set()))).toEqual([3, 3.5, 4, 4.5])
    expect(bits(visibleLadderRungs(six, six[4]!.key, new Set()))).toEqual([3.5, 4, 4.5, 6])
    expect(bits(visibleLadderRungs(six, six[5]!.key, new Set()))).toEqual([3.5, 4, 4.5, 6])
  })

  it("defaults to the smallest four rungs when no rung is recommended", () => {
    expect(bits(visibleLadderRungs(six, null, new Set()))).toEqual([2, 3, 3.5, 4])
  })

  it("keeps an already-owned rung outside the window by dropping the furthest non-anchor rung", () => {
    expect(bits(visibleLadderRungs(six, six[0]!.key, new Set([six[5]!.key])))).toEqual([2, 3, 3.5, 6])
  })

  it("renders at most four columns and a measuring spinner for rungs still being measured", () => {
    const measuring = six.map((item, index) => index === 2 ? { ...item, measuring: true, speed: null } : item)
    const html = renderToStaticMarkup(<QuantLadder quants={measuring} selected={measuring[1]!.key} recommended={measuring[1]!.key} owned={new Set()} limit={null} onSelect={() => {}} />)
    expect(html).toContain("grid-template-columns:repeat(4, minmax(0, 1fr))")
    expect(html).toContain('aria-label="Measuring"')
    expect(html).not.toContain("6-bit")
  })

  it("never renders 0-bit when a quant has an unknown bit width", () => {
    const unknownBits: QuantView = {
      key: "hf:org/repo/custom.gguf",
      name: "CUSTOM",
      publisher: "org",
      repository: "org/repo",
      bits: 0,
      fits: true,
      measuring: false,
      speed: 42,
      phrase: null,
      sizeBytes: 4e9,
      downloadBytes: 4e9,
      drafterBytes: 0,
      drafterTarget: null,
      placement: [{ kind: "Device", bytes: 4e9 }],
      servedContext: 32_768,
      quality: null,
      missingDiskBytes: 0,
    }
    const offered: OfferedModel = {
      key: "custom",
      catalog: false,
      name: "Custom Model",
      lab: undefined,
      country: null,
      fineTuneOf: null,
      drafter: null,
      contextLimit: 32_768,
      quants: [unknownBits],
      recommended: unknownBits.key,
      checking: false,
    }
    const listHtml = renderToStaticMarkup(<QuantList quants={[unknownBits]} selected={null} recommended={null} owned={new Set()} onSelect={() => {}} />)
    const ladderHtml = renderToStaticMarkup(<QuantLadder quants={[unknownBits]} selected={unknownBits.key} recommended={unknownBits.key} owned={new Set()} limit={null} onSelect={() => {}} />)
    const barHtml = renderToStaticMarkup(<PlacementBar quant={unknownBits} />)
    const cardHtml = renderToStaticMarkup(
      <ExpandingModel model={offered} expanded onToggle={() => {}} quantChoice={null} onQuantChoice={() => {}} owned={new Set()} />,
    )
    for (const html of [listHtml, ladderHtml, barHtml, cardHtml]) {
      expect(html).not.toContain("0-bit")
    }
    expect(listHtml).toContain('aria-label="org CUSTOM"')
    expect(ladderHtml).toContain("CUSTOM")
    expect(barHtml).toContain("How this model runs on your machine")
  })

  it("hides the catalog drafter badge when the chosen quant withholds its drafter", () => {
    const withheld: QuantView = {
      key: "qwen3.5-4b:gguf:q6",
      bits: 6,
      publisher: "unsloth",
      name: "UD-Q6_K_XL",
      repository: "unsloth/Qwen3.5-4B-MTP-GGUF",
      sizeBytes: 4_940_000_000,
      downloadBytes: 4_940_000_000,
      drafterBytes: 0,
      drafterWithheld: true,
      drafterTarget: null,
      fits: true,
      measuring: false,
      speed: 11,
      phrase: "Fits in memory",
      placement: [{ kind: "Unified", bytes: 4_940_000_000 }],
      servedContext: 262_144,
      quality: null,
      missingDiskBytes: 0,
    }
    const offered: OfferedModel = {
      key: "qwen3.5-4b",
      catalog: true,
      name: "Qwen3.5 4B",
      lab: undefined,
      country: "CN",
      fineTuneOf: null,
      drafter: { method: "DFlash", sizeBytes: 1_100_000_000 },
      contextLimit: 262_144,
      quants: [withheld],
      recommended: withheld.key,
      checking: false,
    }
    const html = renderToStaticMarkup(
      <ExpandingModel model={offered} expanded={false} onToggle={() => {}} quantChoice={null} onQuantChoice={() => {}} owned={new Set()} />,
    )
    expect(html).not.toContain("DFlash")
  })

  it("renders Hugging Face repository rows with downloads first, uploader, quant count, no context, and Best label on the right", () => {
    const q4: QuantView = {
      key: "hf:unsloth/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf",
      bits: 4,
      publisher: "unsloth",
      name: "UD-Q4_K_XL",
      repository: "unsloth/gemma-4-26B-A4B-it-GGUF",
      sizeBytes: 17.4e9,
      downloadBytes: 17.4e9,
      drafterBytes: 0,
      drafterTarget: null,
      fits: true,
      measuring: false,
      speed: 62,
      phrase: "Fits in memory",
      placement: [{ kind: "Unified", bytes: 17.4e9 }],
      servedContext: 131_072,
      quality: null,
      missingDiskBytes: 0,
    }
    const offered: OfferedModel = {
      key: "unsloth/gemma-4-26B-A4B-it-GGUF",
      catalog: false,
      name: "Gemma 4 26B A4B It",
      lab: undefined,
      country: null,
      fineTuneOf: null,
      drafter: null,
      contextLimit: 131_072,
      quants: [q4],
      recommended: q4.key,
      checking: false,
      downloads: 16_760_416,
      uploader: "unsloth",
      quantCount: 25,
    }
    const html = renderToStaticMarkup(
      <ExpandingModel model={offered} expanded={false} onToggle={() => {}} quantChoice={null} onQuantChoice={() => {}} owned={new Set()} />,
    )
    expect(html).toContain("16.8M / mo")
    expect(html).toContain("unsloth")
    expect(html).toContain("25 quants")
    expect(html).toContain("Best: UD-Q4_K_XL")
    expect(html).not.toContain("4-bit")
    expect(html).not.toContain("context")
    expect(html.indexOf("16.8M / mo")).toBeLessThan(html.indexOf("unsloth"))
    expect(html.indexOf("unsloth")).toBeLessThan(html.indexOf("25 quants"))
  })

  it("renders featured Magnitude Pick cards with an accent border, badge, quant count, drafter, and Best bit-width label", () => {
    const q4: QuantView = {
      key: "qwen3.8-27b:gguf:q4",
      bits: 4,
      publisher: "unsloth",
      name: "UD-Q4_K_XL",
      repository: "unsloth/Qwen3.8-27B-GGUF",
      sizeBytes: 16.2e9,
      downloadBytes: 16.2e9,
      drafterBytes: 1.1e9,
      drafterTarget: null,
      fits: true,
      measuring: false,
      speed: 42,
      phrase: "Fits in memory",
      placement: [{ kind: "Unified", bytes: 16.2e9 }],
      servedContext: 131_072,
      quality: null,
      missingDiskBytes: 0,
    }
    const q6: QuantView = { ...q4, key: "qwen3.8-27b:gguf:q6", bits: 6, name: "UD-Q6_K_XL", sizeBytes: 22e9, downloadBytes: 22e9 }
    const offered: OfferedModel = {
      key: "qwen3.8-27b",
      catalog: true,
      name: "Qwen3.8 27B",
      lab: undefined,
      country: "CN",
      fineTuneOf: null,
      drafter: { method: "DFlash 2", sizeBytes: 1.1e9 },
      contextLimit: 131_072,
      quants: [q4, q6],
      recommended: q4.key,
      checking: false,
    }
    const html = renderToStaticMarkup(
      <ExpandingModel model={offered} note="Smartest that runs well here" featured expanded={false} onToggle={() => {}} quantChoice={null} onQuantChoice={() => {}} owned={new Set()} />,
    )
    expect(html).toContain("Magnitude Pick")
    expect(html).toContain("border-blue-300")
    expect(html).toContain("Smartest that runs well here")
    expect(html).toContain("2 quants")
    expect(html).toContain("DFlash 2 drafter")
    expect(html).toContain("Best: 4-bit")
  })
})
