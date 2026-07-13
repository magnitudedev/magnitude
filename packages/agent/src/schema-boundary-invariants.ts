import { Option, Schema } from 'effect'
import type { JsonEncoded, JsonEncodedSchema, JsonValue } from '@magnitudedev/utils/schema'
import {
  DisplayState as DisplayStateSchema,
  type DisplayState,
  type ToolMessage,
} from '@magnitudedev/protocol'
import { ToolStateSchema, type ToolStateFromSchema } from './models/tool-state'
import type { ToolHandleFromSchema } from './models/tool-handle-schema'

const displayStateSchema = DisplayStateSchema satisfies JsonEncodedSchema<typeof DisplayStateSchema>
const toolStateSchema = ToolStateSchema satisfies JsonEncodedSchema<typeof ToolStateSchema>

const encodeDisplayState: (state: DisplayState) => JsonEncoded<typeof DisplayStateSchema> = Schema.encodeSync(displayStateSchema)
const encodeToolState: (state: ToolStateFromSchema) => JsonEncoded<typeof ToolStateSchema> = Schema.encodeSync(toolStateSchema)

declare const displayState: DisplayState
declare const toolState: ToolStateFromSchema
declare const toolHandle: ToolHandleFromSchema

const encodedDisplayState: JsonValue = encodeDisplayState(displayState)
const encodedToolState: JsonValue = encodeToolState(toolState)

void encodedDisplayState
void encodedToolState

// @ts-expect-error Encoded display JSON is not Type-side DisplayState.
const displayStateFromEncodedJson: DisplayState = encodedDisplayState
void displayStateFromEncodedJson

if (toolHandle.toolKey === 'spawnWorker') {
  const role = toolHandle.state.role
  void role
}

// @ts-expect-error The keyed tool handle union must be narrowed before accessing tool-specific state.
const unnarrowedRole = toolHandle.state.role
void unnarrowedRole
