import { Schema } from 'effect'

/**
 * ToolDef — the codec-layer representation of a tool declaration.
 *
 * The codec's encode step converts ToolDef[] into the provider's wire tool
 * format (ChatTool[]). Using Schema.Class here because ToolDef values may
 * be constructed from catalog or config data and benefit from validation.
 *
 * parameters — a JSON Schema object describing the tool's input shape.
 *              Left as Schema.Unknown; the codec passes it through verbatim
 *              to the wire ChatTool.function.parameters field.
 */
export class ToolDef extends Schema.Class<ToolDef>('ToolDef')({
  name:        Schema.String,
  description: Schema.String,
  parameters:  Schema.Unknown,
}) {}
