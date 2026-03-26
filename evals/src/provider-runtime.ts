import { createProviderClient, type ProviderClient } from '@magnitudedev/providers'

let _client: ProviderClient | null = null

export async function getEvalProviderClient(): Promise<ProviderClient> {
  if (!_client) {
    _client = await createProviderClient({ slots: ['primary', 'secondary'] as const })
  }
  return _client
}