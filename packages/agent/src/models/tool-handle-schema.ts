import { makeKeyedToolHandleUnionSchemaFromEntries } from '@magnitudedev/harness'
import { ToolStateSchemaEntries } from './tool-state'

export const ToolHandleSchema = makeKeyedToolHandleUnionSchemaFromEntries(ToolStateSchemaEntries)

export type ToolHandleFromSchema = typeof ToolHandleSchema.Type
