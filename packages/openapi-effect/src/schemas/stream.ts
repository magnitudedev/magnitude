import { Schema } from "effect";
import { OpenApiSchema } from "./openapi.js";

export const StreamFraming = Schema.Literal("sse", "ndjson");
export type StreamFraming = typeof StreamFraming.Type;

export const StreamData = Schema.Struct({
  encoding: Schema.Literal("json"),
  schema: OpenApiSchema,
});
export type StreamData = typeof StreamData.Type;

export const StreamTermination = Schema.Union(
  Schema.Struct({
    type: Schema.Literal("sentinel"),
    value: Schema.String,
  }),
  Schema.Struct({ type: Schema.Literal("eof") }),
  Schema.Struct({ type: Schema.Literal("long-lived") })
);
export type StreamTermination = typeof StreamTermination.Type;

export const StreamReconnect = Schema.Union(
  Schema.Struct({ type: Schema.Literal("none") }),
  Schema.Struct({ type: Schema.Literal("last-event-id") })
);
export type StreamReconnect = typeof StreamReconnect.Type;

export const StreamMetadata = Schema.Struct({
  version: Schema.Literal(1),
  responseStatus: Schema.Int.pipe(Schema.between(100, 599)),
  framing: StreamFraming,
  data: StreamData,
  termination: StreamTermination,
  reconnect: StreamReconnect,
});
export type StreamMetadata = typeof StreamMetadata.Type;
