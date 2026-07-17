import { Option, Schema } from "effect";
import { ExtensionKey } from "./config.js";
import { JsonObject, JsonValue } from "./json.js";

const optional = { as: "Option", exact: true } as const;

const Extensions = Schema.Record({ key: ExtensionKey, value: JsonValue });

const extensible = <Fields extends Schema.Struct.Fields>(fields: Fields) =>
  Schema.extend(Schema.Struct(fields), Extensions);

export const ReferenceObject = extensible({
  $ref: Schema.String,
  summary: Schema.optionalWith(Schema.String, optional),
  description: Schema.optionalWith(Schema.String, optional),
});
export type ReferenceObject = typeof ReferenceObject.Type;

const OpenApiSchemaObjectBase = extensible({
  $id: Schema.optionalWith(Schema.String, optional),
  $schema: Schema.optionalWith(Schema.String, optional),
  $ref: Schema.optionalWith(Schema.String, optional),
  $comment: Schema.optionalWith(Schema.String, optional),
  title: Schema.optionalWith(Schema.String, optional),
  description: Schema.optionalWith(Schema.String, optional),
  type: Schema.optionalWith(
    Schema.Union(
      Schema.Literal(
        "null",
        "boolean",
        "object",
        "array",
        "number",
        "string",
        "integer"
      ),
      Schema.Array(
        Schema.Literal(
          "null",
          "boolean",
          "object",
          "array",
          "number",
          "string",
          "integer"
        )
      )
    ),
    optional
  ),
  const: Schema.optionalWith(JsonValue, optional),
  enum: Schema.optionalWith(Schema.Array(JsonValue), optional),
  required: Schema.optionalWith(Schema.Array(Schema.String), optional),
  minProperties: Schema.optionalWith(Schema.NonNegativeInt, optional),
  maxProperties: Schema.optionalWith(Schema.NonNegativeInt, optional),
  minContains: Schema.optionalWith(Schema.NonNegativeInt, optional),
  maxContains: Schema.optionalWith(Schema.NonNegativeInt, optional),
  minItems: Schema.optionalWith(Schema.NonNegativeInt, optional),
  maxItems: Schema.optionalWith(Schema.NonNegativeInt, optional),
  uniqueItems: Schema.optionalWith(Schema.Boolean, optional),
  minLength: Schema.optionalWith(Schema.NonNegativeInt, optional),
  maxLength: Schema.optionalWith(Schema.NonNegativeInt, optional),
  pattern: Schema.optionalWith(Schema.String, optional),
  format: Schema.optionalWith(Schema.String, optional),
  contentEncoding: Schema.optionalWith(Schema.String, optional),
  contentMediaType: Schema.optionalWith(Schema.String, optional),
  minimum: Schema.optionalWith(Schema.Number, optional),
  maximum: Schema.optionalWith(Schema.Number, optional),
  exclusiveMinimum: Schema.optionalWith(Schema.Number, optional),
  exclusiveMaximum: Schema.optionalWith(Schema.Number, optional),
  multipleOf: Schema.optionalWith(
    Schema.Number.pipe(Schema.positive()),
    optional
  ),
  default: Schema.optionalWith(JsonValue, optional),
  examples: Schema.optionalWith(Schema.Array(JsonValue), optional),
  readOnly: Schema.optionalWith(Schema.Boolean, optional),
  writeOnly: Schema.optionalWith(Schema.Boolean, optional),
  deprecated: Schema.optionalWith(Schema.Boolean, optional),
  discriminator: Schema.optionalWith(JsonObject, optional),
  xml: Schema.optionalWith(JsonObject, optional),
  externalDocs: Schema.optionalWith(JsonObject, optional),
});

type OpenApiSchemaBase = typeof OpenApiSchemaObjectBase.Type;
type OpenApiSchemaBaseEncoded = typeof OpenApiSchemaObjectBase.Encoded;

type OpenApiSchemaRecursion<Self> = {
  readonly allOf: Option.Option<readonly Self[]>;
  readonly oneOf: Option.Option<readonly Self[]>;
  readonly anyOf: Option.Option<readonly Self[]>;
  readonly not: Option.Option<Self>;
  readonly if: Option.Option<Self>;
  readonly then: Option.Option<Self>;
  readonly else: Option.Option<Self>;
  readonly properties: Option.Option<Readonly<Record<string, Self>>>;
  readonly patternProperties: Option.Option<Readonly<Record<string, Self>>>;
  readonly additionalProperties: Option.Option<boolean | Self>;
  readonly unevaluatedProperties: Option.Option<boolean | Self>;
  readonly propertyNames: Option.Option<Self>;
  readonly items: Option.Option<Self>;
  readonly prefixItems: Option.Option<readonly Self[]>;
  readonly contains: Option.Option<Self>;
  readonly contentSchema: Option.Option<Self>;
};

type RecursiveOpenApiSchema = boolean | RecursiveOpenApiSchemaObject;
interface RecursiveOpenApiSchemaObject
  extends OpenApiSchemaBase,
    OpenApiSchemaRecursion<RecursiveOpenApiSchema> {}

type OpenApiSchemaRecursionEncoded<Self> = {
  readonly allOf?: readonly Self[];
  readonly oneOf?: readonly Self[];
  readonly anyOf?: readonly Self[];
  readonly not?: Self;
  readonly if?: Self;
  readonly then?: Self;
  readonly else?: Self;
  readonly properties?: Readonly<Record<string, Self>>;
  readonly patternProperties?: Readonly<Record<string, Self>>;
  readonly additionalProperties?: boolean | Self;
  readonly unevaluatedProperties?: boolean | Self;
  readonly propertyNames?: Self;
  readonly items?: Self;
  readonly prefixItems?: readonly Self[];
  readonly contains?: Self;
  readonly contentSchema?: Self;
};

type RecursiveOpenApiSchemaEncoded =
  | boolean
  | RecursiveOpenApiSchemaObjectEncoded;
interface RecursiveOpenApiSchemaObjectEncoded
  extends OpenApiSchemaBaseEncoded,
    OpenApiSchemaRecursionEncoded<RecursiveOpenApiSchemaEncoded> {}

export const OpenApiSchema: Schema.Schema<
  RecursiveOpenApiSchema,
  RecursiveOpenApiSchemaEncoded
> = Schema.suspend(
  (): Schema.Schema<RecursiveOpenApiSchema, RecursiveOpenApiSchemaEncoded> =>
    Schema.Union(Schema.Boolean, OpenApiSchemaObject)
);

export const OpenApiSchemaObject: Schema.Schema<
  RecursiveOpenApiSchemaObject,
  RecursiveOpenApiSchemaObjectEncoded
> = Schema.extend(
  OpenApiSchemaObjectBase,
  Schema.Struct({
    allOf: Schema.optionalWith(Schema.Array(OpenApiSchema), optional),
    oneOf: Schema.optionalWith(Schema.Array(OpenApiSchema), optional),
    anyOf: Schema.optionalWith(Schema.Array(OpenApiSchema), optional),
    not: Schema.optionalWith(OpenApiSchema, optional),
    if: Schema.optionalWith(OpenApiSchema, optional),
    then: Schema.optionalWith(OpenApiSchema, optional),
    else: Schema.optionalWith(OpenApiSchema, optional),
    properties: Schema.optionalWith(
      Schema.Record({ key: Schema.String, value: OpenApiSchema }),
      optional
    ),
    patternProperties: Schema.optionalWith(
      Schema.Record({ key: Schema.String, value: OpenApiSchema }),
      optional
    ),
    additionalProperties: Schema.optionalWith(
      Schema.Union(Schema.Boolean, OpenApiSchema),
      optional
    ),
    unevaluatedProperties: Schema.optionalWith(
      Schema.Union(Schema.Boolean, OpenApiSchema),
      optional
    ),
    propertyNames: Schema.optionalWith(OpenApiSchema, optional),
    items: Schema.optionalWith(OpenApiSchema, optional),
    prefixItems: Schema.optionalWith(Schema.Array(OpenApiSchema), optional),
    contains: Schema.optionalWith(OpenApiSchema, optional),
    contentSchema: Schema.optionalWith(OpenApiSchema, optional),
  })
);

export type OpenApiSchema = typeof OpenApiSchema.Type;
export type OpenApiSchemaEncoded = typeof OpenApiSchema.Encoded;
export type OpenApiSchemaObject = typeof OpenApiSchemaObject.Type;
export type OpenApiSchemaObjectEncoded = typeof OpenApiSchemaObject.Encoded;

export const ExampleObject = extensible({
  summary: Schema.optionalWith(Schema.String, optional),
  description: Schema.optionalWith(Schema.String, optional),
  value: Schema.optionalWith(JsonValue, optional),
  externalValue: Schema.optionalWith(Schema.String, optional),
});

export const MediaTypeObject = extensible({
  schema: Schema.optionalWith(OpenApiSchema, optional),
  example: Schema.optionalWith(JsonValue, optional),
  examples: Schema.optionalWith(
    Schema.Record({
      key: Schema.String,
      value: Schema.Union(ExampleObject, ReferenceObject),
    }),
    optional
  ),
  encoding: Schema.optionalWith(JsonObject, optional),
});
export type MediaTypeObject = typeof MediaTypeObject.Type;

export const ContentObject = Schema.Record({
  key: Schema.String,
  value: MediaTypeObject,
});
export type ContentObject = typeof ContentObject.Type;

export const ParameterObject = extensible({
  name: Schema.String,
  in: Schema.Literal("query", "header", "path", "cookie"),
  description: Schema.optionalWith(Schema.String, optional),
  required: Schema.optionalWith(Schema.Boolean, optional),
  deprecated: Schema.optionalWith(Schema.Boolean, optional),
  allowEmptyValue: Schema.optionalWith(Schema.Boolean, optional),
  style: Schema.optionalWith(Schema.String, optional),
  explode: Schema.optionalWith(Schema.Boolean, optional),
  allowReserved: Schema.optionalWith(Schema.Boolean, optional),
  schema: Schema.optionalWith(OpenApiSchema, optional),
  example: Schema.optionalWith(JsonValue, optional),
  examples: Schema.optionalWith(
    Schema.Record({
      key: Schema.String,
      value: Schema.Union(ExampleObject, ReferenceObject),
    }),
    optional
  ),
  content: Schema.optionalWith(ContentObject, optional),
});
export type ParameterObject = typeof ParameterObject.Type;

export const HeaderObject = extensible({
  description: Schema.optionalWith(Schema.String, optional),
  required: Schema.optionalWith(Schema.Boolean, optional),
  deprecated: Schema.optionalWith(Schema.Boolean, optional),
  style: Schema.optionalWith(Schema.String, optional),
  explode: Schema.optionalWith(Schema.Boolean, optional),
  schema: Schema.optionalWith(OpenApiSchema, optional),
  example: Schema.optionalWith(JsonValue, optional),
  examples: Schema.optionalWith(
    Schema.Record({
      key: Schema.String,
      value: Schema.Union(ExampleObject, ReferenceObject),
    }),
    optional
  ),
  content: Schema.optionalWith(ContentObject, optional),
});

export const RequestBodyObject = extensible({
  description: Schema.optionalWith(Schema.String, optional),
  content: ContentObject,
  required: Schema.optionalWith(Schema.Boolean, optional),
});
export type RequestBodyObject = typeof RequestBodyObject.Type;

export const ResponseObject = extensible({
  description: Schema.String,
  headers: Schema.optionalWith(
    Schema.Record({
      key: Schema.String,
      value: Schema.Union(HeaderObject, ReferenceObject),
    }),
    optional
  ),
  content: Schema.optionalWith(ContentObject, optional),
  links: Schema.optionalWith(JsonObject, optional),
});
export type ResponseObject = typeof ResponseObject.Type;

export const ExternalDocumentationObject = extensible({
  description: Schema.optionalWith(Schema.String, optional),
  url: Schema.String,
});

export const ServerVariableObject = extensible({
  enum: Schema.optionalWith(Schema.Array(Schema.String), optional),
  default: Schema.String,
  description: Schema.optionalWith(Schema.String, optional),
});

export const ServerObject = extensible({
  url: Schema.String,
  description: Schema.optionalWith(Schema.String, optional),
  variables: Schema.optionalWith(
    Schema.Record({ key: Schema.String, value: ServerVariableObject }),
    optional
  ),
});

export const SecurityRequirementObject = Schema.Record({
  key: Schema.String,
  value: Schema.Array(Schema.String),
});

export const OperationObject = extensible({
  tags: Schema.optionalWith(Schema.Array(Schema.String), optional),
  summary: Schema.optionalWith(Schema.String, optional),
  description: Schema.optionalWith(Schema.String, optional),
  externalDocs: Schema.optionalWith(ExternalDocumentationObject, optional),
  operationId: Schema.optionalWith(Schema.String, optional),
  parameters: Schema.optionalWith(
    Schema.Array(Schema.Union(ParameterObject, ReferenceObject)),
    optional
  ),
  requestBody: Schema.optionalWith(
    Schema.Union(RequestBodyObject, ReferenceObject),
    optional
  ),
  responses: Schema.Record({
    key: Schema.String,
    value: Schema.Union(ResponseObject, ReferenceObject),
  }),
  callbacks: Schema.optionalWith(JsonObject, optional),
  deprecated: Schema.optionalWith(Schema.Boolean, optional),
  security: Schema.optionalWith(
    Schema.Array(SecurityRequirementObject),
    optional
  ),
  servers: Schema.optionalWith(Schema.Array(ServerObject), optional),
});
export type OperationObject = typeof OperationObject.Type;

export const PathItemObject = extensible({
  $ref: Schema.optionalWith(Schema.String, optional),
  summary: Schema.optionalWith(Schema.String, optional),
  description: Schema.optionalWith(Schema.String, optional),
  get: Schema.optionalWith(OperationObject, optional),
  put: Schema.optionalWith(OperationObject, optional),
  post: Schema.optionalWith(OperationObject, optional),
  delete: Schema.optionalWith(OperationObject, optional),
  options: Schema.optionalWith(OperationObject, optional),
  head: Schema.optionalWith(OperationObject, optional),
  patch: Schema.optionalWith(OperationObject, optional),
  trace: Schema.optionalWith(OperationObject, optional),
  servers: Schema.optionalWith(Schema.Array(ServerObject), optional),
  parameters: Schema.optionalWith(
    Schema.Array(Schema.Union(ParameterObject, ReferenceObject)),
    optional
  ),
});
export type PathItemObject = typeof PathItemObject.Type;

export const InfoObject = extensible({
  title: Schema.String,
  summary: Schema.optionalWith(Schema.String, optional),
  description: Schema.optionalWith(Schema.String, optional),
  termsOfService: Schema.optionalWith(Schema.String, optional),
  contact: Schema.optionalWith(JsonObject, optional),
  license: Schema.optionalWith(JsonObject, optional),
  version: Schema.String,
});

export const TagObject = extensible({
  name: Schema.String,
  description: Schema.optionalWith(Schema.String, optional),
  externalDocs: Schema.optionalWith(ExternalDocumentationObject, optional),
});

export const ComponentsObject = extensible({
  schemas: Schema.optionalWith(
    Schema.Record({ key: Schema.String, value: OpenApiSchema }),
    optional
  ),
  responses: Schema.optionalWith(
    Schema.Record({
      key: Schema.String,
      value: Schema.Union(ResponseObject, ReferenceObject),
    }),
    optional
  ),
  parameters: Schema.optionalWith(
    Schema.Record({
      key: Schema.String,
      value: Schema.Union(ParameterObject, ReferenceObject),
    }),
    optional
  ),
  examples: Schema.optionalWith(
    Schema.Record({
      key: Schema.String,
      value: Schema.Union(ExampleObject, ReferenceObject),
    }),
    optional
  ),
  requestBodies: Schema.optionalWith(
    Schema.Record({
      key: Schema.String,
      value: Schema.Union(RequestBodyObject, ReferenceObject),
    }),
    optional
  ),
  headers: Schema.optionalWith(
    Schema.Record({
      key: Schema.String,
      value: Schema.Union(HeaderObject, ReferenceObject),
    }),
    optional
  ),
  securitySchemes: Schema.optionalWith(JsonObject, optional),
  links: Schema.optionalWith(JsonObject, optional),
  callbacks: Schema.optionalWith(JsonObject, optional),
  pathItems: Schema.optionalWith(
    Schema.Record({
      key: Schema.String,
      value: Schema.Union(PathItemObject, ReferenceObject),
    }),
    optional
  ),
});
export type ComponentsObject = typeof ComponentsObject.Type;

export const OpenApiDocument = extensible({
  openapi: Schema.String.pipe(Schema.pattern(/^3\.1(?:\.|$)/)),
  info: InfoObject,
  jsonSchemaDialect: Schema.optionalWith(Schema.String, optional),
  servers: Schema.optionalWith(Schema.Array(ServerObject), optional),
  paths: Schema.optionalWith(
    Schema.Record({ key: Schema.String, value: PathItemObject }),
    optional
  ),
  webhooks: Schema.optionalWith(
    Schema.Record({
      key: Schema.String,
      value: Schema.Union(PathItemObject, ReferenceObject),
    }),
    optional
  ),
  components: Schema.optionalWith(ComponentsObject, optional),
  security: Schema.optionalWith(
    Schema.Array(SecurityRequirementObject),
    optional
  ),
  tags: Schema.optionalWith(Schema.Array(TagObject), optional),
  externalDocs: Schema.optionalWith(ExternalDocumentationObject, optional),
});
export type OpenApiDocument = typeof OpenApiDocument.Type;
