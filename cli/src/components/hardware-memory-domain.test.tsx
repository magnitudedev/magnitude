import { renderToStaticMarkup } from 'react-dom/server'
import { expect, test } from 'vitest'
import type { HardwareMemoryDomainView } from '@magnitudedev/client-common'
import { defaultCliThemes } from '../utils/theme'
import { HardwareMemoryDomain } from './hardware-memory-domain'

const theme = defaultCliThemes.dark

const domain: HardwareMemoryDomainView = {
  id: 'unified-memory' as HardwareMemoryDomainView['id'],
  label: 'Apple M4 Max · Unified memory',
  kind: 'UnifiedMemory',
  totalBytes: 64 * 1024 ** 3,
  usedBytes: 7 * 1024 ** 3,
  fixedBytes: 0,
  kvCacheBytes: 0,
  systemAndAppsBytes: 7 * 1024 ** 3,
  freeBytes: 57 * 1024 ** 3,
  status: 'complete',
  notice: null,
  participatesInModelServing: true,
}

test('gives complete memory legend rows an explicit theme foreground', () => {
  const html = renderToStaticMarkup(<HardwareMemoryDomain domain={domain} />)

  for (const label of ['Weights', 'KV cache', 'System &amp; apps', 'Free']) {
    expect(html).toMatch(new RegExp(`<text style="fg:${theme.text.body}">[^<]*<span[^>]*>[^<]*</span> ${label}`))
  }
})
