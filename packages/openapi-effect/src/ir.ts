import { Chunk, Data, HashMap, Option } from "effect";
import type { JsonPrimitive } from "./schemas/json.js";

export interface SourceLocation {
  readonly pointer: string;
}

export interface StringConstraints {
  readonly minLength: Option.Option<number>;
  readonly maxLength: Option.Option<number>;
  readonly pattern: Option.Option<string>;
}

export interface NumberConstraints {
  readonly minimum: Option.Option<number>;
  readonly maximum: Option.Option<number>;
  readonly exclusiveMinimum: Option.Option<number>;
  readonly exclusiveMaximum: Option.Option<number>;
  readonly multipleOf: Option.Option<number>;
}

export interface ArrayConstraints {
  readonly minItems: Option.Option<number>;
  readonly maxItems: Option.Option<number>;
}

export interface ObjectProperty {
  readonly name: string;
  readonly schema: SchemaNode;
  readonly required: boolean;
  readonly readOnly: boolean;
  readonly writeOnly: boolean;
  readonly source: SourceLocation;
}

export type AdditionalProperties = Data.TaggedEnum<{
  Allowed: Record<never, never>;
  Forbidden: Record<never, never>;
  Typed: { readonly schema: SchemaNode };
}>;
export const AdditionalProperties = Data.taggedEnum<AdditionalProperties>();

export type SchemaNode = Data.TaggedEnum<{
  JsonValue: { readonly source: SourceLocation };
  Never: { readonly source: SourceLocation };
  Null: { readonly source: SourceLocation };
  Literal: { readonly value: JsonPrimitive; readonly source: SourceLocation };
  Enum: {
    readonly values: readonly JsonPrimitive[];
    readonly source: SourceLocation;
  };
  String: {
    readonly format: Option.Option<string>;
    readonly constraints: StringConstraints;
    readonly source: SourceLocation;
  };
  Number: {
    readonly integer: boolean;
    readonly constraints: NumberConstraints;
    readonly source: SourceLocation;
  };
  Boolean: { readonly source: SourceLocation };
  Array: {
    readonly items: SchemaNode;
    readonly constraints: ArrayConstraints;
    readonly source: SourceLocation;
  };
  Object: {
    readonly properties: readonly ObjectProperty[];
    readonly additionalProperties: AdditionalProperties;
    readonly source: SourceLocation;
  };
  Union: {
    readonly mode: "AnyOf" | "OneOf";
    readonly members: readonly SchemaNode[];
    readonly source: SourceLocation;
  };
  Ref: {
    readonly target: string;
    readonly source: SourceLocation;
  };
}>;
export const SchemaNode = Data.taggedEnum<SchemaNode>();

export interface ComponentSchema {
  readonly sourceName: string;
  readonly name: string;
  readonly schema: SchemaNode;
  readonly source: SourceLocation;
}

export type HttpMethod =
  | "GET"
  | "POST"
  | "PUT"
  | "PATCH"
  | "DELETE"
  | "HEAD"
  | "OPTIONS";
export type ParameterLocation = "Path" | "Query" | "Header";

export interface OperationParameter {
  readonly name: string;
  readonly location: ParameterLocation;
  readonly required: boolean;
  readonly schema: SchemaNode;
  readonly source: SourceLocation;
}

export type MediaType =
  | "application/json"
  | "text/plain"
  | "application/octet-stream";

export interface OperationBody {
  readonly mediaType: MediaType;
  readonly schema: SchemaNode;
  readonly required: boolean;
  readonly source: SourceLocation;
}

export interface OperationResponse {
  readonly status: number;
  readonly success: boolean;
  readonly mediaType: Option.Option<MediaType>;
  readonly schema: Option.Option<SchemaNode>;
  readonly source: SourceLocation;
}

interface OperationCommon {
  readonly operationId: string;
  readonly name: string;
  readonly group: string;
  readonly groupName: string;
  readonly method: HttpMethod;
  readonly path: string;
  readonly parameters: readonly OperationParameter[];
  readonly body: Option.Option<OperationBody>;
  readonly source: SourceLocation;
}

export type Operation = Data.TaggedEnum<{
  Http: OperationCommon & {
    readonly responses: readonly OperationResponse[];
  };
  Ndjson: OperationCommon & {
    readonly eventSchema: SchemaNode;
    readonly mediaType: string;
    readonly errors: readonly OperationResponse[];
  };
}>;
export const Operation = Data.taggedEnum<Operation>();

export interface ProtocolIr {
  readonly title: string;
  readonly version: string;
  readonly components: HashMap.HashMap<string, ComponentSchema>;
  readonly operations: Chunk.Chunk<Operation>;
}
