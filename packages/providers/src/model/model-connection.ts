import { Data } from 'effect'
import type { AuthInfo } from '../types'

export type ModelConnection = Data.TaggedEnum<{
  Baml: {
    readonly auth: AuthInfo | null
  }
}>

export const ModelConnection = Data.taggedEnum<ModelConnection>()