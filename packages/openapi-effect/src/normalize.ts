import { Chunk, Context, Effect, HashMap, Option, Ref, Schema } from "effect";
import { Diagnostic, OpenApiSemanticError } from "./errors.js";
import {
  AdditionalProperties,
  Operation,
  SchemaNode,
  StreamReconnect,
  StreamTermination,
  type ComponentSchema,
  type HttpMethod,
  type MediaType,
  type ObjectProperty,
  type OperationBody,
  type OperationParameter,
  type OperationResponse,
  type ParameterLocation,
  type ProtocolIr,
  type SchemaNode as SchemaNodeType,
} from "./ir.js";
import type { OpenApiEffectConfig } from "./schemas/config.js";
import {
  StreamMetadata,
  type StreamMetadata as StreamMetadataType,
} from "./schemas/stream.js";
import {
  ReferenceObject,
  type ComponentsObject,
  type OpenApiDocument,
  type OpenApiSchema as OpenApiSchemaType,
  type OpenApiSchemaObject,
  type OperationObject,
  type ParameterObject,
  type PathItemObject,
  type RequestBodyObject,
  type ResponseObject,
} from "./schemas/openapi.js";
import type { JsonPrimitive, JsonValue } from "./schemas/json.js";

interface NormalizationContextShape {
  readonly config: OpenApiEffectConfig;
  readonly document: OpenApiDocument;
  readonly componentNames: HashMap.HashMap<string, string>;
  readonly diagnostics: Ref.Ref<Chunk.Chunk<Diagnostic>>;
}

class NormalizationContext extends Context.Tag(
  "@magnitudedev/openapi-effect/NormalizationContext"
)<NormalizationContext, NormalizationContextShape>() {}

const pointerEscape = (value: string): string =>
  value.replaceAll("~", "~0").replaceAll("/", "~1");
const pointerChild = (pointer: string, child: string | number): string =>
  `${pointer}/${pointerEscape(String(child))}`;

const report = (
  context: NormalizationContextShape,
  code: string,
  pointer: string,
  message: string
): Effect.Effect<void> =>
  Ref.update(
    context.diagnostics,
    Chunk.append(Diagnostic.make({ code, pointer, message, related: [] }))
  );

const pascalIdentifier = (value: string): string => {
  const words = Option.getOrElse(
    Option.fromNullable(value.match(/[A-Za-z0-9]+/g)),
    () => [] as const
  );
  const joined = words
    .map((word) => `${word.charAt(0).toUpperCase()}${word.slice(1)}`)
    .join("");
  const identifier = joined.length > 0 ? joined : "Generated";
  return /^[0-9]/.test(identifier) ? `_${identifier}` : identifier;
};

const camelIdentifier = (value: string): string => {
  const pascal = pascalIdentifier(value);
  return `${pascal.charAt(0).toLowerCase()}${pascal.slice(1)}`;
};

const decodeRefToken = (value: string): string =>
  value.replaceAll("~1", "/").replaceAll("~0", "~");

const directRefName = (
  ref: string,
  category: string
): Option.Option<string> => {
  const expression = new RegExp(`^#/components/${category}/([^/]+)$`);
  return Option.fromNullable(expression.exec(ref)).pipe(
    Option.flatMap((match) => Option.fromNullable(match[1])),
    Option.map(decodeRefToken)
  );
};

const isPrimitive = (value: JsonValue): value is JsonPrimitive =>
  value === null ||
  typeof value === "string" ||
  typeof value === "number" ||
  typeof value === "boolean";

const schemaHasSemanticRefSibling = (schema: OpenApiSchemaObject): boolean =>
  Option.isSome(schema.type) ||
  Option.isSome(schema.const) ||
  Option.isSome(schema.enum) ||
  Option.isSome(schema.allOf) ||
  Option.isSome(schema.oneOf) ||
  Option.isSome(schema.anyOf) ||
  Option.isSome(schema.not) ||
  Option.isSome(schema.if) ||
  Option.isSome(schema.then) ||
  Option.isSome(schema.else) ||
  Option.isSome(schema.properties) ||
  Option.isSome(schema.patternProperties) ||
  Option.isSome(schema.additionalProperties) ||
  Option.isSome(schema.unevaluatedProperties) ||
  Option.isSome(schema.propertyNames) ||
  Option.isSome(schema.items) ||
  Option.isSome(schema.prefixItems) ||
  Option.isSome(schema.contains) ||
  Option.isSome(schema.contentSchema);

const keywordWhenPresent = <A>(
  keyword: string,
  value: Option.Option<A>
): readonly string[] => (Option.isSome(value) ? [keyword] : []);

const unsupportedSchemaKeywords = (
  schema: OpenApiSchemaObject
): ReadonlyArray<string> => [
  ...keywordWhenPresent("allOf", schema.allOf),
  ...keywordWhenPresent("not", schema.not),
  ...keywordWhenPresent("if", schema.if),
  ...keywordWhenPresent("then", schema.then),
  ...keywordWhenPresent("else", schema.else),
  ...keywordWhenPresent("patternProperties", schema.patternProperties),
  ...keywordWhenPresent("unevaluatedProperties", schema.unevaluatedProperties),
  ...keywordWhenPresent("propertyNames", schema.propertyNames),
  ...keywordWhenPresent("prefixItems", schema.prefixItems),
  ...keywordWhenPresent("contains", schema.contains),
  ...keywordWhenPresent("contentSchema", schema.contentSchema),
];

const normalizeSchema = (
  schema: OpenApiSchemaType,
  pointer: string
): Effect.Effect<SchemaNodeType, never, NormalizationContext> =>
  Effect.gen(function* () {
    const context = yield* NormalizationContext;
    const source = { pointer };
    if (schema === true) return SchemaNode.JsonValue({ source });
    if (schema === false) return SchemaNode.Never({ source });

    if (Option.isSome(schema.$ref)) {
      const target = directRefName(schema.$ref.value, "schemas");
      if (Option.isNone(target)) {
        yield* report(
          context,
          "ref.unsupported",
          pointerChild(pointer, "$ref"),
          "Only direct local component schema references are supported"
        );
        return SchemaNode.Never({ source });
      }
      if (!HashMap.has(context.componentNames, target.value)) {
        yield* report(
          context,
          "ref.missing",
          pointerChild(pointer, "$ref"),
          `Unknown component schema ${target.value}`
        );
      }
      if (schemaHasSemanticRefSibling(schema)) {
        yield* report(
          context,
          "ref.semantic-siblings",
          pointer,
          "Semantic $ref siblings are not supported"
        );
      }
      return SchemaNode.Ref({ target: target.value, source });
    }

    for (const keyword of unsupportedSchemaKeywords(schema)) {
      yield* report(
        context,
        `schema.${keyword}-unsupported`,
        pointerChild(pointer, keyword),
        `${keyword} is not supported`
      );
    }

    if (Option.isSome(schema.const)) {
      if (!isPrimitive(schema.const.value)) {
        yield* report(
          context,
          "schema.const-nonprimitive",
          pointerChild(pointer, "const"),
          "Only primitive const values are supported"
        );
        return SchemaNode.Never({ source });
      }
      return SchemaNode.Literal({ value: schema.const.value, source });
    }

    if (Option.isSome(schema.enum)) {
      const primitiveValues = schema.enum.value.filter(isPrimitive);
      if (
        primitiveValues.length === 0 ||
        primitiveValues.length !== schema.enum.value.length
      ) {
        yield* report(
          context,
          "schema.enum-nonprimitive",
          pointerChild(pointer, "enum"),
          "enum must contain only primitive JSON values"
        );
      }
      return primitiveValues.length === 0
        ? SchemaNode.Never({ source })
        : SchemaNode.Enum({ values: primitiveValues, source });
    }

    const union: Option.Option<
      readonly ["AnyOf" | "OneOf", readonly OpenApiSchemaType[]]
    > = Option.match(schema.oneOf, {
      onNone: () =>
        Option.map(schema.anyOf, (members) => ["AnyOf", members] as const),
      onSome: (members) => Option.some(["OneOf", members] as const),
    });
    if (Option.isSome(union)) {
      const [mode, values] = union.value;
      const members = yield* Effect.forEach(values, (member, index) =>
        normalizeSchema(
          member,
          pointerChild(
            pointerChild(pointer, mode === "OneOf" ? "oneOf" : "anyOf"),
            index
          )
        )
      );
      if (members.length === 0) {
        yield* report(
          context,
          "schema.empty-union",
          pointer,
          `${mode} cannot be empty`
        );
        return SchemaNode.Never({ source });
      }
      if (mode === "OneOf" && !isProvablyExclusive(members)) {
        yield* report(
          context,
          "schema.oneOf-overlap",
          pointerChild(pointer, "oneOf"),
          "oneOf branches are not provably exclusive"
        );
      }
      return SchemaNode.Union({ mode, members, source });
    }

    if (Option.isSome(schema.type) && Array.isArray(schema.type.value)) {
      const distinct = [...new Set(schema.type.value)];
      const nonNull = distinct.filter((entry) => entry !== "null");
      const nullableMember = Option.fromNullable(nonNull[0]);
      if (
        distinct.length === 2 &&
        nonNull.length === 1 &&
        Option.isSome(nullableMember)
      ) {
        const member = yield* normalizeSchema(
          { ...schema, type: Option.some(nullableMember.value) },
          pointer
        );
        return SchemaNode.Union({
          mode: "AnyOf",
          members: [member, SchemaNode.Null({ source })],
          source,
        });
      }
      yield* report(
        context,
        "schema.type-array-unsupported",
        pointerChild(pointer, "type"),
        "Only a single type plus null is supported"
      );
      return SchemaNode.Never({ source });
    }

    const effectiveType = Option.orElse(schema.type, () =>
      Option.isSome(schema.properties) ||
      Option.isSome(schema.additionalProperties)
        ? Option.some("object" as const)
        : Option.none()
    );
    if (Option.isNone(effectiveType)) return SchemaNode.JsonValue({ source });
    if (Array.isArray(effectiveType.value)) {
      return SchemaNode.Never({ source });
    }
    switch (effectiveType.value) {
      case "null":
        return SchemaNode.Null({ source });
      case "boolean":
        return SchemaNode.Boolean({ source });
      case "string":
        return SchemaNode.String({
          format: schema.format,
          constraints: {
            minLength: schema.minLength,
            maxLength: schema.maxLength,
            pattern: schema.pattern,
          },
          source,
        });
      case "number":
      case "integer":
        return SchemaNode.Number({
          integer: effectiveType.value === "integer",
          constraints: {
            minimum: schema.minimum,
            maximum: schema.maximum,
            exclusiveMinimum: schema.exclusiveMinimum,
            exclusiveMaximum: schema.exclusiveMaximum,
            multipleOf: schema.multipleOf,
          },
          source,
        });
      case "array": {
        if (Option.isNone(schema.items)) {
          yield* report(
            context,
            "schema.array-items-required",
            pointerChild(pointer, "items"),
            "Array schemas require items"
          );
        }
        const items = yield* normalizeSchema(
          Option.getOrElse(schema.items, () => false),
          pointerChild(pointer, "items")
        );
        return SchemaNode.Array({
          items,
          constraints: {
            minItems: schema.minItems,
            maxItems: schema.maxItems,
          },
          source,
        });
      }
      case "object": {
        const required = new Set(
          Option.getOrElse(schema.required, () => [] as const)
        );
        const properties = yield* Effect.forEach(
          Object.entries(Option.getOrElse(schema.properties, () => ({}))).sort(
            ([left], [right]) => left.localeCompare(right)
          ),
          ([name, propertySchema]): Effect.Effect<
            ObjectProperty,
            never,
            NormalizationContext
          > =>
            Effect.map(
              normalizeSchema(
                propertySchema,
                pointerChild(pointerChild(pointer, "properties"), name)
              ),
              (normalized) => ({
                name,
                schema: normalized,
                required: required.has(name),
                readOnly:
                  propertySchema !== true &&
                  propertySchema !== false &&
                  Option.contains(propertySchema.readOnly, true),
                writeOnly:
                  propertySchema !== true &&
                  propertySchema !== false &&
                  Option.contains(propertySchema.writeOnly, true),
                source: {
                  pointer: pointerChild(
                    pointerChild(pointer, "properties"),
                    name
                  ),
                },
              })
            )
        );
        for (const property of properties) {
          if (property.readOnly || property.writeOnly) {
            yield* report(
              context,
              "schema.directionality-unsupported",
              property.source.pointer,
              "readOnly/writeOnly properties require direction-specific generated schemas and are not supported yet"
            );
          }
        }
        for (const requiredName of [...required].sort()) {
          if (
            !(requiredName in Option.getOrElse(schema.properties, () => ({})))
          ) {
            yield* report(
              context,
              "schema.required-property-missing",
              pointerChild(pointer, "required"),
              `Required property ${requiredName} is not declared`
            );
          }
        }
        const additionalProperties = Option.isNone(schema.additionalProperties)
          ? AdditionalProperties.Allowed()
          : schema.additionalProperties.value === true
          ? AdditionalProperties.Allowed()
          : schema.additionalProperties.value === false
          ? AdditionalProperties.Forbidden()
          : AdditionalProperties.Typed({
              schema: yield* normalizeSchema(
                schema.additionalProperties.value,
                pointerChild(pointer, "additionalProperties")
              ),
            });
        return SchemaNode.Object({ properties, additionalProperties, source });
      }
    }
    yield* report(
      context,
      "schema.type-unsupported",
      pointerChild(pointer, "type"),
      `Unsupported schema type ${effectiveType.value}`
    );
    return SchemaNode.Never({ source });
  });

const primitiveCategory = (schema: SchemaNodeType): Option.Option<string> =>
  SchemaNode.$match(schema, {
    JsonValue: () => Option.none(),
    Never: () => Option.none(),
    Null: () => Option.some("null"),
    Literal: ({ value }) => Option.some(value === null ? "null" : typeof value),
    Enum: ({ values }) => {
      const categories = new Set(
        values.map((value) => (value === null ? "null" : typeof value))
      );
      return categories.size === 1
        ? Option.fromNullable([...categories][0])
        : Option.none();
    },
    String: () => Option.some("string"),
    Number: () => Option.some("number"),
    Boolean: () => Option.some("boolean"),
    Array: () => Option.some("array"),
    Object: () => Option.none(),
    Union: () => Option.none(),
    Ref: () => Option.none(),
  });

const discriminator = (
  schema: SchemaNodeType
): Option.Option<readonly [string, JsonPrimitive]> =>
  SchemaNode.$match(schema, {
    Object: ({ properties }) => {
      return Option.fromNullable(
        properties.find(
          (candidate) =>
            candidate.required &&
            (candidate.schema._tag === "Literal" ||
              (candidate.schema._tag === "Enum" &&
                candidate.schema.values.length === 1))
        )
      ).pipe(
        Option.flatMap((property) =>
          property.schema._tag === "Literal"
            ? Option.some([property.name, property.schema.value] as const)
            : property.schema._tag === "Enum"
            ? Option.map(
                Option.fromNullable(property.schema.values[0]),
                (value) => [property.name, value] as const
              )
            : Option.none()
        )
      );
    },
    JsonValue: () => Option.none(),
    Never: () => Option.none(),
    Null: () => Option.none(),
    Literal: () => Option.none(),
    Enum: () => Option.none(),
    String: () => Option.none(),
    Number: () => Option.none(),
    Boolean: () => Option.none(),
    Array: () => Option.none(),
    Union: () => Option.none(),
    Ref: () => Option.none(),
  });

const isProvablyExclusive = (members: readonly SchemaNodeType[]): boolean => {
  const categories = members.map(primitiveCategory);
  if (
    categories.every(Option.isSome) &&
    new Set(categories.map(({ value }) => value)).size === categories.length
  )
    return true;
  const discriminators = members.map(discriminator);
  if (!discriminators.every(Option.isSome)) return false;
  return (
    new Set(discriminators.map(({ value }) => value[0])).size === 1 &&
    new Set(discriminators.map(({ value }) => JSON.stringify(value[1])))
      .size === discriminators.length
  );
};

const resolveParameter = (
  value: ParameterObject | ReferenceObject,
  pointer: string
): Effect.Effect<Option.Option<ParameterObject>, never, NormalizationContext> =>
  Effect.gen(function* () {
    const context = yield* NormalizationContext;
    if (Schema.is(ReferenceObject)(value)) {
      const name = directRefName(value.$ref, "parameters");
      if (Option.isNone(name)) {
        yield* report(
          context,
          "ref.unsupported",
          pointer,
          "Expected a local components.parameters reference"
        );
        return Option.none();
      }
      const resolved = Option.flatMap(
        context.document.components,
        (components) =>
          Option.flatMap(components.parameters, (parameters) =>
            Option.fromNullable(parameters[name.value])
          )
      );
      if (
        Option.isNone(resolved) ||
        Schema.is(ReferenceObject)(resolved.value)
      ) {
        yield* report(
          context,
          "ref.missing-or-chained",
          pointer,
          `Unable to resolve parameter ${name.value}`
        );
        return Option.none();
      }
      return Option.some(resolved.value);
    }
    return Option.some(value);
  });

const normalizeParameter = (
  value: ParameterObject | ReferenceObject,
  pointer: string
): Effect.Effect<
  Option.Option<OperationParameter>,
  never,
  NormalizationContext
> =>
  Effect.gen(function* () {
    const context = yield* NormalizationContext;
    const resolved = yield* resolveParameter(value, pointer);
    if (Option.isNone(resolved)) return Option.none();
    const parameter = resolved.value;
    if (parameter.in === "cookie") {
      yield* report(
        context,
        "parameter.cookie-unsupported",
        pointerChild(pointer, "in"),
        "Cookie parameters are not supported"
      );
      return Option.none();
    }
    if (Option.isNone(parameter.schema)) {
      yield* report(
        context,
        "parameter.schema-required",
        pointer,
        "Parameter schema is required; content parameters are unsupported"
      );
    }
    const location: ParameterLocation =
      parameter.in === "path"
        ? "Path"
        : parameter.in === "query"
        ? "Query"
        : "Header";
    const required =
      location === "Path" || Option.contains(parameter.required, true);
    if (location === "Path" && !Option.contains(parameter.required, true)) {
      yield* report(
        context,
        "parameter.path-required",
        pointer,
        "Path parameters must set required: true"
      );
    }
    const expectedStyle = location === "Query" ? "form" : "simple";
    const expectedExplode = location === "Query";
    if (
      Option.isSome(parameter.style) &&
      parameter.style.value !== expectedStyle
    ) {
      yield* report(
        context,
        "parameter.style-unsupported",
        pointerChild(pointer, "style"),
        `Only ${expectedStyle} style is supported here`
      );
    }
    if (
      Option.isSome(parameter.explode) &&
      parameter.explode.value !== expectedExplode
    ) {
      yield* report(
        context,
        "parameter.explode-unsupported",
        pointerChild(pointer, "explode"),
        `Only explode=${expectedExplode} is supported here`
      );
    }
    return Option.some({
      name: parameter.name,
      location,
      required,
      schema: yield* normalizeSchema(
        Option.getOrElse(parameter.schema, () => false),
        pointerChild(pointer, "schema")
      ),
      source: { pointer },
    });
  });

const resolveRequestBody = (
  value: RequestBodyObject | ReferenceObject,
  pointer: string
): Effect.Effect<
  Option.Option<RequestBodyObject>,
  never,
  NormalizationContext
> =>
  Effect.gen(function* () {
    const context = yield* NormalizationContext;
    if (!Schema.is(ReferenceObject)(value)) return Option.some(value);
    const name = directRefName(value.$ref, "requestBodies");
    const resolved = Option.flatMap(name, (componentName) =>
      Option.flatMap(context.document.components, (components) =>
        Option.flatMap(components.requestBodies, (requestBodies) =>
          Option.fromNullable(requestBodies[componentName])
        )
      )
    );
    if (Option.isNone(resolved) || Schema.is(ReferenceObject)(resolved.value)) {
      yield* report(
        context,
        "ref.missing-or-chained",
        pointer,
        "Unable to resolve request body reference"
      );
      return Option.none();
    }
    return Option.some(resolved.value);
  });

const isSupportedMediaType = (value: string): value is MediaType =>
  value === "application/json" ||
  value === "text/plain" ||
  value === "application/octet-stream";

const selectMediaType = <A>(
  content: Readonly<Record<string, A>>,
  pointer: string
): Effect.Effect<
  Option.Option<readonly [MediaType, A]>,
  never,
  NormalizationContext
> =>
  Effect.gen(function* () {
    const context = yield* NormalizationContext;
    const entries = Object.entries(content);
    const supported = entries.filter((entry): entry is [MediaType, A] =>
      isSupportedMediaType(entry[0])
    );
    if (entries.length !== 1 || supported.length !== 1) {
      yield* report(
        context,
        "media-type.unsupported",
        pointer,
        "Exactly one JSON, text, or byte media type is required"
      );
      return Option.none();
    }
    return Option.fromNullable(supported[0]);
  });

const normalizeBody = (
  value: RequestBodyObject | ReferenceObject,
  pointer: string
): Effect.Effect<Option.Option<OperationBody>, never, NormalizationContext> =>
  Effect.gen(function* () {
    const context = yield* NormalizationContext;
    const resolved = yield* resolveRequestBody(value, pointer);
    if (Option.isNone(resolved)) return Option.none();
    const selected = yield* selectMediaType(
      resolved.value.content,
      pointerChild(pointer, "content")
    );
    if (Option.isNone(selected)) return Option.none();
    const [mediaType, media] = selected.value;
    if (Option.isSome(media.encoding)) {
      yield* report(
        context,
        "body.encoding-unsupported",
        pointerChild(pointerChild(pointer, "content"), mediaType),
        "Per-property content encodings are not supported"
      );
    }
    if (Option.isNone(media.schema)) {
      yield* report(
        context,
        "body.schema-required",
        pointer,
        "Request body media type requires a schema"
      );
    }
    return Option.some({
      mediaType,
      schema: yield* normalizeSchema(
        Option.getOrElse(media.schema, () => false),
        pointerChild(pointerChild(pointer, "content"), mediaType)
      ),
      required: Option.contains(resolved.value.required, true),
      source: { pointer },
    });
  });

const resolveResponse = (
  value: ResponseObject | ReferenceObject,
  pointer: string
): Effect.Effect<Option.Option<ResponseObject>, never, NormalizationContext> =>
  Effect.gen(function* () {
    const context = yield* NormalizationContext;
    if (!Schema.is(ReferenceObject)(value)) return Option.some(value);
    const name = directRefName(value.$ref, "responses");
    const resolved = Option.flatMap(name, (componentName) =>
      Option.flatMap(context.document.components, (components) =>
        Option.flatMap(components.responses, (responses) =>
          Option.fromNullable(responses[componentName])
        )
      )
    );
    if (Option.isNone(resolved) || Schema.is(ReferenceObject)(resolved.value)) {
      yield* report(
        context,
        "ref.missing-or-chained",
        pointer,
        "Unable to resolve response reference"
      );
      return Option.none();
    }
    return Option.some(resolved.value);
  });

const normalizeResponses = (
  responses: OperationObject["responses"],
  pointer: string,
  requireSuccess = true
): Effect.Effect<readonly OperationResponse[], never, NormalizationContext> =>
  Effect.gen(function* () {
    const context = yield* NormalizationContext;
    const normalized = yield* Effect.forEach(
      Object.entries(responses).sort(([left], [right]) =>
        left.localeCompare(right)
      ),
      ([statusKey, value]) =>
        Effect.gen(function* () {
          const responsePointer = pointerChild(pointer, statusKey);
          if (!/^\d{3}$/.test(statusKey)) {
            yield* report(
              context,
              "response.status-unsupported",
              responsePointer,
              "Only exact three-digit response statuses are supported"
            );
            return Option.none<OperationResponse>();
          }
          const status = Number(statusKey);
          const response = yield* resolveResponse(value, responsePointer);
          if (Option.isNone(response)) return Option.none<OperationResponse>();
          if (Option.isSome(response.value.headers)) {
            yield* report(
              context,
              "response.headers-unsupported",
              pointerChild(responsePointer, "headers"),
              "Response headers are not supported yet"
            );
          }
          if (Option.isSome(response.value.links)) {
            yield* report(
              context,
              "response.links-unsupported",
              pointerChild(responsePointer, "links"),
              "Response links are not supported"
            );
          }
          if (
            Option.isNone(response.value.content) ||
            Object.keys(response.value.content.value).length === 0
          ) {
            return Option.some({
              status,
              success: status >= 200 && status < 300,
              mediaType: Option.none(),
              schema: Option.none(),
              source: { pointer: responsePointer },
            });
          }
          const selected = yield* selectMediaType(
            response.value.content.value,
            pointerChild(responsePointer, "content")
          );
          if (Option.isNone(selected)) return Option.none<OperationResponse>();
          const [mediaType, media] = selected.value;
          if (Option.isSome(media.encoding)) {
            yield* report(
              context,
              "response.encoding-unsupported",
              pointerChild(pointerChild(responsePointer, "content"), mediaType),
              "Per-property content encodings are not supported"
            );
          }
          if (Option.isNone(media.schema))
            yield* report(
              context,
              "response.schema-required",
              responsePointer,
              "Response media type requires a schema"
            );
          return Option.some({
            status,
            success: status >= 200 && status < 300,
            mediaType: Option.some(mediaType),
            schema: Option.some(
              yield* normalizeSchema(
                Option.getOrElse(media.schema, () => false),
                pointerChild(
                  pointerChild(responsePointer, "content"),
                  mediaType
                )
              )
            ),
            source: { pointer: responsePointer },
          });
        })
    );
    const values = normalized.flatMap(Option.toArray);
    if (requireSuccess && !values.some(({ success }) => success)) {
      yield* report(
        context,
        "response.success-required",
        pointer,
        "At least one exact 2xx response is required"
      );
    }
    return values;
  });

const pathParameterNames = (path: string): readonly string[] =>
  [...path.matchAll(/\{([^{}]+)\}/g)]
    .flatMap((match) => Option.toArray(Option.fromNullable(match[1])))
    .sort();

const normalizeOperation = (
  method: HttpMethod,
  path: string,
  pathItem: PathItemObject,
  operation: OperationObject,
  pointer: string
): Effect.Effect<Option.Option<Operation>, never, NormalizationContext> =>
  Effect.gen(function* () {
    const context = yield* NormalizationContext;
    if (
      Option.isNone(operation.operationId) ||
      operation.operationId.value.length === 0
    ) {
      yield* report(
        context,
        "operation.id-required",
        pointerChild(pointer, "operationId"),
        "A non-empty operationId is required"
      );
      return Option.none();
    }
    const tag = Option.flatMap(operation.tags, (tags) =>
      Option.fromNullable(tags[0])
    );
    if (Option.isNone(tag))
      yield* report(
        context,
        "operation.tag-required",
        pointerChild(pointer, "tags"),
        "At least one operation tag is required"
      );
    const group = Option.getOrElse(tag, () => "default");
    if (Option.isSome(operation.callbacks)) {
      yield* report(
        context,
        "operation.callbacks-unsupported",
        pointerChild(pointer, "callbacks"),
        "Callbacks are not supported"
      );
    }
    if (Option.isSome(operation.security)) {
      yield* report(
        context,
        "operation.security-unsupported",
        pointerChild(pointer, "security"),
        "Operation security requirements are not supported yet"
      );
    }
    if (Option.isSome(operation.servers)) {
      yield* report(
        context,
        "operation.servers-unsupported",
        pointerChild(pointer, "servers"),
        "Operation-specific servers are not supported"
      );
    }
    const configuredExtensions: ReadonlySet<string> = new Set(
      context.config.transports.map(({ extension }) => extension)
    );
    for (const extension of Object.keys(operation).filter((key) =>
      key.startsWith("x-")
    )) {
      if (!configuredExtensions.has(extension)) {
        yield* report(
          context,
          "operation.extension-unrecognized",
          pointerChild(pointer, extension),
          `No compiler plugin claims ${extension}`
        );
      }
    }
    const parameterValues = [
      ...Option.getOrElse(pathItem.parameters, () => [] as const),
      ...Option.getOrElse(operation.parameters, () => [] as const),
    ];
    const parameterOptions = yield* Effect.forEach(
      parameterValues,
      (parameter, index) =>
        normalizeParameter(
          parameter,
          pointerChild(pointerChild(pointer, "parameters"), index)
        )
    );
    const parameters = parameterOptions.flatMap(Option.toArray);
    const seen = new Set<string>();
    for (const parameter of parameters) {
      const key = `${parameter.location}:${parameter.name.toLowerCase()}`;
      if (seen.has(key))
        yield* report(
          context,
          "parameter.duplicate",
          parameter.source.pointer,
          `Duplicate parameter ${key}`
        );
      seen.add(key);
    }
    const declaredPathNames = parameters
      .filter(({ location }) => location === "Path")
      .map(({ name }) => name)
      .sort();
    const templateNames = pathParameterNames(path);
    if (
      declaredPathNames.length !== templateNames.length ||
      declaredPathNames.some((name, index) => name !== templateNames[index])
    ) {
      yield* report(
        context,
        "parameter.path-mismatch",
        pointer,
        "Path template and declared path parameters do not match"
      );
    }
    const body = Option.isNone(operation.requestBody)
      ? Option.none<OperationBody>()
      : yield* normalizeBody(
          operation.requestBody.value,
          pointerChild(pointer, "requestBody")
        );
    if (
      Option.isSome(body) &&
      (method === "GET" || method === "HEAD" || method === "OPTIONS")
    ) {
      yield* report(
        context,
        "body.method-unsupported",
        pointerChild(pointer, "requestBody"),
        `${method} request bodies cannot be represented by Effect HttpApi`
      );
    }

    const matchedTransports = context.config.transports.filter((transport) =>
      Option.isSome(Option.fromNullable(operation[transport.extension]))
    );
    if (matchedTransports.length > 1) {
      yield* report(
        context,
        "transport.ambiguous",
        pointer,
        "Operation matches more than one custom transport"
      );
      return Option.none();
    }
    const common = {
      operationId: operation.operationId.value,
      name: camelIdentifier(operation.operationId.value),
      group,
      groupName: camelIdentifier(group),
      method,
      path,
      parameters,
      body,
      source: { pointer },
    };
    const matchedTransport = Option.fromNullable(matchedTransports[0]);
    if (matchedTransports.length === 1 && Option.isSome(matchedTransport)) {
      const transport = matchedTransport.value;
      const extensionPointer = pointerChild(pointer, transport.extension);
      const metadata = yield* Schema.decodeUnknown(StreamMetadata)(
        operation[transport.extension],
        { onExcessProperty: "error" }
      ).pipe(
        Effect.map(Option.some),
        Effect.catchTag("ParseError", () =>
          Effect.as(
            report(
              context,
              "transport.metadata-invalid",
              extensionPointer,
              `${transport.extension} does not match the configured stream metadata schema`
            ),
            Option.none<StreamMetadataType>()
          )
        )
      );
      if (Option.isNone(metadata)) return Option.none();
      const stream = metadata.value;
      if (stream.responseStatus < 200 || stream.responseStatus >= 300) {
        yield* report(
          context,
          "transport.success-status-required",
          extensionPointer,
          "Stream responseStatus must be an exact 2xx status"
        );
      }
      const statusKey = String(stream.responseStatus);
      const declaredResponse = Option.fromNullable(
        operation.responses[statusKey]
      );
      const expectedMediaType =
        stream.framing === "sse" ? "text/event-stream" : "application/x-ndjson";
      if (Option.isNone(declaredResponse)) {
        yield* report(
          context,
          "transport.response-missing",
          pointerChild(pointer, "responses"),
          `Stream response ${statusKey} is not documented`
        );
      } else {
        const resolvedResponse = yield* resolveResponse(
          declaredResponse.value,
          pointerChild(pointerChild(pointer, "responses"), statusKey)
        );
        const media = Option.flatMap(resolvedResponse, (response) =>
          Option.flatMap(response.content, (content) =>
            Option.fromNullable(content[expectedMediaType])
          )
        );
        if (Option.isNone(media)) {
          yield* report(
            context,
            "transport.media-type-required",
            pointerChild(pointerChild(pointer, "responses"), statusKey),
            `Expected ${expectedMediaType} response content`
          );
        } else {
          const responseBody = yield* normalizeSchema(
            Option.getOrElse(media.value.schema, () => false),
            pointerChild(
              pointerChild(
                pointerChild(pointerChild(pointer, "responses"), statusKey),
                "content"
              ),
              expectedMediaType
            )
          );
          if (responseBody._tag !== "String") {
            yield* report(
              context,
              "transport.response-body-string-required",
              pointerChild(pointerChild(pointer, "responses"), statusKey),
              "The standard OpenAPI stream response body must be a string"
            );
          }
        }
      }
      const eventSchema = yield* normalizeSchema(
        stream.data.schema,
        pointerChild(pointerChild(extensionPointer, "data"), "schema")
      );
      if (eventSchema._tag !== "Ref") {
        yield* report(
          context,
          "transport.event-schema-component-required",
          pointer,
          "Streaming event schemas must be direct component references"
        );
      }
      const errorResponses = Object.fromEntries(
        Object.entries(operation.responses).filter(
          ([status]) => /^\d{3}$/.test(status) && Number(status) >= 300
        )
      );
      return Option.some(
        Operation.Stream({
          ...common,
          framing: stream.framing === "sse" ? "Sse" : "Ndjson",
          eventSchema,
          mediaType: expectedMediaType,
          responseStatus: stream.responseStatus,
          termination:
            stream.termination.type === "sentinel"
              ? StreamTermination.Sentinel({
                  value: stream.termination.value,
                })
              : stream.termination.type === "eof"
              ? StreamTermination.Eof()
              : StreamTermination.LongLived(),
          reconnect:
            stream.reconnect.type === "last-event-id"
              ? StreamReconnect.LastEventId()
              : StreamReconnect.None(),
          errors: yield* normalizeResponses(
            errorResponses,
            pointerChild(pointer, "responses"),
            false
          ),
        })
      );
    }
    return Option.some(
      Operation.Http({
        ...common,
        responses: yield* normalizeResponses(
          operation.responses,
          pointerChild(pointer, "responses")
        ),
      })
    );
  });

const methodEntries = (
  pathItem: PathItemObject
): readonly (readonly [
  HttpMethod,
  Option.Option<OperationObject>,
  string
])[] => [
  ["GET", pathItem.get, "get"],
  ["PUT", pathItem.put, "put"],
  ["POST", pathItem.post, "post"],
  ["DELETE", pathItem.delete, "delete"],
  ["OPTIONS", pathItem.options, "options"],
  ["HEAD", pathItem.head, "head"],
  ["PATCH", pathItem.patch, "patch"],
];

const normalizeComponents = (
  components: Option.Option<ComponentsObject>
): Effect.Effect<
  HashMap.HashMap<string, ComponentSchema>,
  never,
  NormalizationContext
> =>
  Effect.gen(function* () {
    const context = yield* NormalizationContext;
    const schemas = Option.flatMap(components, ({ schemas }) => schemas);
    const entries = Object.entries(Option.getOrElse(schemas, () => ({}))).sort(
      ([left], [right]) => left.localeCompare(right)
    );
    const normalized = yield* Effect.forEach(
      entries,
      ([sourceName, schema]) => {
        const pointer = `#/components/schemas/${pointerEscape(sourceName)}`;
        return Effect.map(
          normalizeSchema(schema, pointer),
          (value): readonly [string, ComponentSchema] => [
            sourceName,
            {
              sourceName,
              name: HashMap.get(context.componentNames, sourceName).pipe(
                Option.getOrElse(() => pascalIdentifier(sourceName))
              ),
              schema: value,
              source: { pointer },
            },
          ]
        );
      }
    );
    return HashMap.fromIterable(normalized);
  });

const wireSchemaIssue = (
  schema: SchemaNodeType,
  location: ParameterLocation,
  components: HashMap.HashMap<string, ComponentSchema>,
  visited: ReadonlySet<string> = new Set()
): Option.Option<string> =>
  SchemaNode.$match(schema, {
    String: () => Option.none(),
    Number: () => Option.none(),
    Boolean: () => Option.none(),
    Literal: ({ value }) =>
      value === null
        ? Option.some("null cannot be serialized as an HTTP parameter")
        : Option.none(),
    Enum: ({ values }) => {
      const kinds = new Set(
        values.map((value) => (value === null ? "null" : typeof value))
      );
      return kinds.size === 1 && !kinds.has("null")
        ? Option.none()
        : Option.some(
            "HTTP parameter enums must contain one non-null primitive kind"
          );
    },
    Array: ({ items }) =>
      location === "Query"
        ? wireSchemaIssue(items, location, components, visited)
        : Option.some("Only query parameters may be arrays"),
    Union: ({ members }) =>
      Option.fromNullable(
        members.flatMap((member) =>
          Option.toArray(wireSchemaIssue(member, location, components, visited))
        )[0]
      ),
    Ref: ({ target }) => {
      if (visited.has(target))
        return Option.some("Recursive schemas cannot be HTTP parameters");
      return HashMap.get(components, target).pipe(
        Option.match({
          onNone: () => Option.some(`Missing component schema ${target}`),
          onSome: (component) =>
            wireSchemaIssue(
              component.schema,
              location,
              components,
              new Set([...visited, target])
            ),
        })
      );
    },
    JsonValue: () =>
      Option.some("Unconstrained JSON values cannot be HTTP parameters"),
    Never: () => Option.some("Never schemas cannot be HTTP parameters"),
    Null: () => Option.some("null cannot be serialized as an HTTP parameter"),
    Object: () =>
      Option.some(
        "Object parameters require an unsupported serialization style"
      ),
  });

export const normalizeOpenApi = (
  document: OpenApiDocument,
  config: OpenApiEffectConfig
): Effect.Effect<ProtocolIr, OpenApiSemanticError> =>
  Effect.gen(function* () {
    const diagnostics = yield* Ref.make(Chunk.empty<Diagnostic>());
    const componentSchemas = Option.flatMap(
      document.components,
      ({ schemas }) => schemas
    );
    const componentEntries = Object.keys(
      Option.getOrElse(componentSchemas, () => ({}))
    )
      .sort()
      .map((sourceName) => [sourceName, pascalIdentifier(sourceName)] as const);
    const componentNames = HashMap.fromIterable(componentEntries);
    const context: NormalizationContextShape = {
      config,
      document,
      componentNames,
      diagnostics,
    };
    const normalize = Effect.gen(function* () {
      const context = yield* NormalizationContext;
      if (Option.isSome(document.webhooks)) {
        yield* report(
          context,
          "document.webhooks-unsupported",
          "#/webhooks",
          "Webhooks are not supported"
        );
      }
      if (Option.isSome(document.security)) {
        yield* report(
          context,
          "document.security-unsupported",
          "#/security",
          "Document security requirements are not supported yet"
        );
      }
      if (Option.isSome(document.servers)) {
        yield* report(
          context,
          "document.servers-unsupported",
          "#/servers",
          "Document servers are not used by generated clients"
        );
      }
      if (
        Option.exists(document.components, ({ securitySchemes }) =>
          Option.isSome(securitySchemes)
        )
      ) {
        yield* report(
          context,
          "components.security-schemes-unsupported",
          "#/components/securitySchemes",
          "Security schemes are not supported yet"
        );
      }
      if (
        Option.exists(document.components, ({ links }) => Option.isSome(links))
      ) {
        yield* report(
          context,
          "components.links-unsupported",
          "#/components/links",
          "Component links are not supported"
        );
      }
      if (
        Option.exists(document.components, ({ callbacks }) =>
          Option.isSome(callbacks)
        )
      ) {
        yield* report(
          context,
          "components.callbacks-unsupported",
          "#/components/callbacks",
          "Component callbacks are not supported"
        );
      }
      if (
        Option.exists(document.components, ({ pathItems }) =>
          Option.isSome(pathItems)
        )
      ) {
        yield* report(
          context,
          "components.path-items-unsupported",
          "#/components/pathItems",
          "Component path items are not supported"
        );
      }
      const generatedNames = new Map<string, string>();
      for (const [sourceName, generatedName] of componentEntries) {
        const previous = Option.fromNullable(generatedNames.get(generatedName));
        if (Option.isSome(previous)) {
          yield* report(
            context,
            "name.component-collision",
            `#/components/schemas/${pointerEscape(sourceName)}`,
            `Generated name ${generatedName} collides with ${previous.value}`
          );
        }
        generatedNames.set(generatedName, sourceName);
      }
      const components = yield* normalizeComponents(document.components);
      const operations = yield* Effect.forEach(
        Object.entries(Option.getOrElse(document.paths, () => ({}))).sort(
          ([left], [right]) => left.localeCompare(right)
        ),
        ([path, pathItem]) =>
          Effect.gen(function* () {
            const pathPointer = `#/paths/${pointerEscape(path)}`;
            if (!path.startsWith("/"))
              yield* report(
                context,
                "path.invalid",
                pathPointer,
                "Paths must start with /"
              );
            if (Option.isSome(pathItem.$ref))
              yield* report(
                context,
                "path.ref-unsupported",
                pointerChild(pathPointer, "$ref"),
                "Referenced path items are not supported"
              );
            if (Option.isSome(pathItem.servers)) {
              yield* report(
                context,
                "path.servers-unsupported",
                pointerChild(pathPointer, "servers"),
                "Path-specific servers are not supported"
              );
            }
            if (Option.isSome(pathItem.trace))
              yield* report(
                context,
                "method.trace-unsupported",
                pointerChild(pathPointer, "trace"),
                "TRACE is not supported by Effect HttpApi"
              );
            return yield* Effect.forEach(
              methodEntries(pathItem),
              ([method, operation, methodName]) =>
                Option.isNone(operation)
                  ? Effect.succeed(Option.none<Operation>())
                  : normalizeOperation(
                      method,
                      path,
                      pathItem,
                      operation.value,
                      pointerChild(pathPointer, methodName)
                    )
            );
          })
      );
      const operationValues = operations.flat().flatMap(Option.toArray);
      for (const operation of operationValues) {
        for (const parameter of operation.parameters) {
          const issue = wireSchemaIssue(
            parameter.schema,
            parameter.location,
            components
          );
          if (Option.isSome(issue))
            yield* report(
              context,
              "parameter.schema-unsupported",
              parameter.source.pointer,
              issue.value
            );
        }
      }
      const operationNames = new Map<string, string>();
      const groupNames = new Map<string, string>();
      for (const operation of operationValues) {
        const previousOperation = Option.fromNullable(
          operationNames.get(operation.name)
        );
        if (Option.isSome(previousOperation))
          yield* report(
            context,
            "name.operation-collision",
            operation.source.pointer,
            `Operation name ${operation.name} collides with ${previousOperation.value}`
          );
        operationNames.set(operation.name, operation.operationId);
        const previousGroup = Option.fromNullable(
          groupNames.get(operation.groupName)
        );
        if (
          Option.isSome(previousGroup) &&
          previousGroup.value !== operation.group
        ) {
          yield* report(
            context,
            "name.group-collision",
            operation.source.pointer,
            `Group name ${operation.groupName} collides with ${previousGroup.value}`
          );
        }
        groupNames.set(operation.groupName, operation.group);
      }
      const issues = yield* Ref.get(diagnostics);
      if (Chunk.isNonEmpty(issues)) {
        return yield* new OpenApiSemanticError({
          diagnostics: [...issues].sort(
            (left, right) =>
              left.pointer.localeCompare(right.pointer) ||
              left.code.localeCompare(right.code) ||
              left.message.localeCompare(right.message)
          ),
        });
      }
      return {
        title: document.info.title,
        version: document.info.version,
        components,
        operations: Chunk.fromIterable(
          operationValues.sort((left, right) =>
            left.operationId.localeCompare(right.operationId)
          )
        ),
      };
    });
    return yield* normalize.pipe(
      Effect.provideService(NormalizationContext, context)
    );
  });
