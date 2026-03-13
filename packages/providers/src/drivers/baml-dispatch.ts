import { b, Collector } from '@magnitudedev/llm-core'

// Derive types from the actual BAML client
type BamlClient = typeof b
type StreamClient = BamlClient['stream']
type StreamRequestClient = BamlClient['streamRequest']
type ParseClient = BamlClient['parse']

// Derive from Collector if possible, otherwise define minimal interface
export interface CollectorCall {
  httpRequest?: { body: { json(): unknown } }
  httpResponse?: { body: { json(): unknown } }
}

export interface CollectorStreamCall extends CollectorCall {
  sseResponses(): Array<{ json?(): unknown }>
}

// Runtime-validated dispatch helpers
// The `as keyof` casts are guarded by the `in` check
export function bamlStream(name: string, args: readonly unknown[], opts: object): AsyncIterable<string> {
  if (!(name in b.stream)) throw new Error(`Unknown BAML stream function: ${name}`)
  const fn = b.stream[name as keyof StreamClient] as (...args: unknown[]) => AsyncIterable<string>
  return fn.call(b.stream, ...args, opts)
}

export function bamlCall(name: string, args: readonly unknown[], opts: object): Promise<unknown> {
  if (!(name in b)) throw new Error(`Unknown BAML function: ${name}`)
  const fn = b[name as keyof BamlClient] as (...args: unknown[]) => Promise<unknown>
  return fn.call(b, ...args, opts)
}

export interface StreamRequestResult {
  body: { json(): Record<string, unknown> }
}

export function bamlStreamRequest(
  name: string,
  args: readonly unknown[],
  opts: object,
): Promise<StreamRequestResult> {
  if (!(name in b.streamRequest)) throw new Error(`Unknown BAML streamRequest function: ${name}`)
  const fn = b.streamRequest[name as keyof StreamRequestClient] as (...args: unknown[]) => Promise<StreamRequestResult>
  return fn.call(b.streamRequest, ...args, opts)
}

export function bamlParse(name: string, text: string): unknown {
  if (!(name in b.parse)) throw new Error(`Unknown BAML parse function: ${name}`)
  const fn = b.parse[name as keyof ParseClient] as (text: string) => unknown
  return fn.call(b.parse, text)
}

export function getLastCollectorCall(collector: Collector): CollectorCall | undefined {
  return collector.last?.calls.at(-1) as CollectorCall | undefined
}

export function getLastCollectorStreamCall(collector: Collector): CollectorStreamCall | undefined {
  const call = collector.last?.calls.at(-1)
  if (call && 'sseResponses' in call && typeof call.sseResponses === 'function') {
    return call as CollectorStreamCall
  }
  return undefined
}