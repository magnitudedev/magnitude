import { b } from '@magnitudedev/llm-core'

type BamlClient = typeof b
type BamlStreamClient = BamlClient['stream']

export type BamlFunctionName = keyof BamlClient & string
export type BamlStreamFunctionName = keyof BamlStreamClient & string

export type BamlResult<K extends BamlFunctionName> =
  BamlClient[K] extends (...args: any[]) => Promise<infer R> ? R : never

export type BamlStreamReturn<K extends BamlStreamFunctionName> =
  BamlStreamClient[K] extends (...args: any[]) => infer R ? R : never