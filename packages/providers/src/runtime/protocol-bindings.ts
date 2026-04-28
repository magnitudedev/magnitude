/**
 * ProtocolBindings — maps (providerProtocol, paradigm) → driverId + codecId.
 *
 * Used by NativeModelResolverLive to pick the right driver + codec for a given
 * provider model. Only the native paradigm is wired; xml-act and completions
 * paradigms are future work (or legacy BAML path).
 */

import { Context, Layer } from 'effect'

export interface ProtocolBinding {
  readonly driverId: string
  readonly codecId: string
}

export interface ProtocolBindingsShape {
  readonly lookup: (
    protocol: string,
    paradigm: 'xml-act' | 'native' | 'completions',
  ) => ProtocolBinding | null
}

export class ProtocolBindings extends Context.Tag('ProtocolBindings')<
  ProtocolBindings,
  ProtocolBindingsShape
>() {}

const BINDINGS: Record<string, Record<string, ProtocolBinding>> = {
  'openai-generic': {
    native: { driverId: 'openai-chat-completions', codecId: 'native-chat-completions' },
  },
  'openai': {
    native: { driverId: 'openai-chat-completions', codecId: 'native-chat-completions' },
  },
}

export const ProtocolBindingsLive = Layer.succeed(ProtocolBindings, {
  lookup: (protocol, paradigm) => BINDINGS[protocol]?.[paradigm] ?? null,
})
