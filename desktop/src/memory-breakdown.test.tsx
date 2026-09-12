import { renderToStaticMarkup } from "react-dom/server"
import { expect, it } from "vitest"
import { LocalInferenceMemoryDomainIdSchema } from "@magnitudedev/sdk"
import type { HardwareMemoryDomainView } from "@magnitudedev/client-common"
import { MemoryDomain } from "./memory-breakdown"

const domain: HardwareMemoryDomainView = {
  id: LocalInferenceMemoryDomainIdSchema.make("system"), label: "Unified memory", kind: "UnifiedMemory",
  totalBytes: 16 * 1024 ** 3, usedBytes: 8 * 1024 ** 3,
  modelBytes: 3 * 1024 ** 3, overheadBytes: 1024 ** 3, fixedBytes: 4 * 1024 ** 3,
  kvCacheBytes: 2 * 1024 ** 3, systemAndAppsBytes: 2 * 1024 ** 3, freeBytes: 8 * 1024 ** 3,
  status: "complete", notice: null, participatesInModelServing: true,
}
it("renders separate accessible memory categories with exact byte evidence", () => {
  const html = renderToStaticMarkup(<MemoryDomain domain={domain} />)
  for (const label of ["Model weights", "KV cache", "Engine overhead", "System &amp; apps", "Free"]) expect(html).toContain(`data-memory-category="${label}"`)
  expect(html).toContain('data-memory-category="Model weights" data-bytes="3221225472"')
  expect(html).not.toContain("Unavailable")
})
it("shows unknown measurements without a zero-valued or partial stacked chart", () => {
  const html = renderToStaticMarkup(<MemoryDomain domain={{ ...domain, modelBytes: null, overheadBytes: null, fixedBytes: null, kvCacheBytes: null, systemAndAppsBytes: null, status: "inconsistent", notice: "Allocation and resident usage could not be reconciled." }} />)
  expect(html.match(/Unavailable/g)).toHaveLength(4)
  expect(html).not.toContain('style="width:')
  expect(html).toContain("could not be reconciled")
  expect(html).not.toContain('data-bytes="0"')
})
