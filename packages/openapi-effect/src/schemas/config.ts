import { Schema } from "effect";

export const ExtensionKey = Schema.TemplateLiteral("x-", Schema.String);
export type ExtensionKey = typeof ExtensionKey.Type;

export class OutputLayout extends Schema.Class<OutputLayout>(
  "OpenApiEffect.OutputLayout"
)({
  schemas: Schema.optionalWith(Schema.String, { default: () => "schemas.ts" }),
  operations: Schema.optionalWith(Schema.String, {
    default: () => "operations.ts",
  }),
  api: Schema.optionalWith(Schema.String, { default: () => "api.ts" }),
  client: Schema.optionalWith(Schema.String, { default: () => "client.ts" }),
  index: Schema.optionalWith(Schema.String, { default: () => "index.ts" }),
  manifest: Schema.optionalWith(Schema.String, {
    default: () => "manifest.json",
  }),
}) {}

export class StreamTransport extends Schema.Class<StreamTransport>(
  "OpenApiEffect.StreamTransport"
)({
  extension: Schema.optionalWith(ExtensionKey, {
    default: () => "x-magnitude-stream" as const,
  }),
}) {}

export const Transport = Schema.Union(StreamTransport);
export type Transport = typeof Transport.Type;

export class OpenApiEffectConfig extends Schema.Class<OpenApiEffectConfig>(
  "OpenApiEffect.Config"
)({
  apiName: Schema.String.pipe(Schema.minLength(1)),
  output: Schema.optionalWith(OutputLayout, {
    default: () => new OutputLayout({}),
  }),
  transports: Schema.optionalWith(Schema.Array(Transport), {
    default: () => [],
  }),
}) {}

export type OpenApiEffectConfigEncoded = typeof OpenApiEffectConfig.Encoded;
