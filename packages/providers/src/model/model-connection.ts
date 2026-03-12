import { Data } from 'effect'
import type { AuthInfo } from '../types'

export type ModelConnection = Data.TaggedEnum<{
  Baml: {
    readonly auth: AuthInfo | null
  }
  Responses: {
    readonly auth: AuthInfo | null
    readonly endpoint: string
    readonly headers: Record<string, string>
  }
}>

export const ModelConnection = Data.taggedEnum<ModelConnection>()