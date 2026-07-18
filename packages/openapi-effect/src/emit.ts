import * as Path from "node:path";
import { Data, Effect, HashMap, Option } from "effect";
import { format } from "prettier";
import { OpenApiEmitError } from "./errors.js";
import {
  AdditionalProperties,
  Operation,
  SchemaNode,
  type ComponentSchema,
  type MediaType,
  type Operation as OperationType,
  type OperationParameter,
  type OperationResponse,
  type ProtocolIr,
  type SchemaNode as SchemaNodeType,
} from "./ir.js";
import type { OpenApiEffectConfig } from "./schemas/config.js";

export class EmittedProject extends Data.Class<{
  readonly files: HashMap.HashMap<string, string>;
}> {}

const quote = (value: string): string => JSON.stringify(value);

const componentName = (
  target: string,
  components: HashMap.HashMap<string, ComponentSchema>
): string =>
  HashMap.get(components, target).pipe(
    Option.map(({ name }) => name),
    Option.getOrElse(() => "MissingComponent")
  );

const literalSchema = (value: string | number | boolean | null): string =>
  value === null ? "S.Null" : `S.Literal(${JSON.stringify(value)})`;

const literalType = (value: string | number | boolean | null): string =>
  value === null ? "null" : JSON.stringify(value);

const withPipe = (base: string, filters: readonly string[]): string =>
  filters.length === 0 ? base : `${base}.pipe(${filters.join(", ")})`;

const emitStringFilters = (
  schema: Extract<SchemaNodeType, { readonly _tag: "String" }>
): readonly string[] => [
  ...Option.toArray(
    Option.map(schema.constraints.minLength, (value) => `S.minLength(${value})`)
  ),
  ...Option.toArray(
    Option.map(schema.constraints.maxLength, (value) => `S.maxLength(${value})`)
  ),
  ...Option.toArray(
    Option.map(
      schema.constraints.pattern,
      (value) => `S.pattern(new RegExp(${quote(value)}))`
    )
  ),
];

const emitNumberFilters = (
  schema: Extract<SchemaNodeType, { readonly _tag: "Number" }>
): readonly string[] => [
  ...(schema.integer ? ["S.int()"] : []),
  ...Option.toArray(
    Option.map(
      schema.constraints.minimum,
      (value) => `S.greaterThanOrEqualTo(${value})`
    )
  ),
  ...Option.toArray(
    Option.map(
      schema.constraints.maximum,
      (value) => `S.lessThanOrEqualTo(${value})`
    )
  ),
  ...Option.toArray(
    Option.map(
      schema.constraints.exclusiveMinimum,
      (value) => `S.greaterThan(${value})`
    )
  ),
  ...Option.toArray(
    Option.map(
      schema.constraints.exclusiveMaximum,
      (value) => `S.lessThan(${value})`
    )
  ),
  ...Option.toArray(
    Option.map(
      schema.constraints.multipleOf,
      (value) => `S.multipleOf(${value})`
    )
  ),
];

const emitSchemaExpression = (
  schema: SchemaNodeType,
  components: HashMap.HashMap<string, ComponentSchema>,
  namespace = ""
): string =>
  SchemaNode.$match(schema, {
    JsonValue: () => `${namespace}JsonValue`,
    Never: () => "S.Never",
    Null: () => "S.Null",
    Literal: ({ value }) => literalSchema(value),
    Enum: ({ values }) => {
      const member = Option.fromNullable(values[0]);
      return values.length === 1 && Option.isSome(member)
        ? literalSchema(member.value)
        : `S.Union(${values.map(literalSchema).join(", ")})`;
    },
    String: (value) => withPipe("S.String", emitStringFilters(value)),
    Number: (value) => withPipe("S.Number", emitNumberFilters(value)),
    Boolean: () => "S.Boolean",
    Array: ({ items, constraints }) =>
      withPipe(
        `S.Array(${emitSchemaExpression(items, components, namespace)})`,
        [
          ...Option.toArray(
            Option.map(constraints.minItems, (value) => `S.minItems(${value})`)
          ),
          ...Option.toArray(
            Option.map(constraints.maxItems, (value) => `S.maxItems(${value})`)
          ),
        ]
      ),
    Object: ({ properties, additionalProperties }) => {
      const fields = properties
        .map(
          (property) =>
            `${quote(property.name)}: ${
              property.required
                ? emitSchemaExpression(property.schema, components, namespace)
                : `S.optionalWith(${emitSchemaExpression(
                    property.schema,
                    components,
                    namespace
                  )}, { exact: true, as: "Option" })`
            }`
        )
        .join(",\n");
      const struct = `S.Struct({${
        fields.length === 0 ? "" : `\n${fields}\n`
      }})`;
      return AdditionalProperties.$match(additionalProperties, {
        Forbidden: () => struct,
        Allowed: () =>
          `S.extend(${struct}, S.Record({ key: S.String, value: JsonValue }))`,
        Typed: ({ schema }) =>
          `S.extend(${struct}, S.Record({ key: S.String, value: ${emitSchemaExpression(
            schema,
            components,
            namespace
          )} }))`,
      });
    },
    Union: ({ members }) => {
      const member = Option.fromNullable(members[0]);
      return members.length === 1 && Option.isSome(member)
        ? emitSchemaExpression(member.value, components, namespace)
        : `S.Union(${members
            .map((value) => emitSchemaExpression(value, components, namespace))
            .join(", ")})`;
    },
    Ref: ({ target }) => {
      const name = componentName(target, components);
      return `S.suspend((): S.Schema<${namespace}${name}, ${namespace}${name}Encoded> => ${namespace}${name})`;
    },
  });

const emitType = (
  schema: SchemaNodeType,
  components: HashMap.HashMap<string, ComponentSchema>,
  decoded: boolean
): string =>
  SchemaNode.$match(schema, {
    JsonValue: () => "JsonValue",
    Never: () => "never",
    Null: () => "null",
    Literal: ({ value }) => literalType(value),
    Enum: ({ values }) => values.map(literalType).join(" | "),
    String: () => "string",
    Number: () => "number",
    Boolean: () => "boolean",
    Array: ({ items }) => `ReadonlyArray<${emitType(items, components, decoded)}>`,
    Object: ({ properties, additionalProperties }) => {
      const fields = properties
        .map(
          (property) =>
            property.required
              ? `readonly ${quote(property.name)}: ${emitType(property.schema, components, decoded)}`
              : decoded
                ? `readonly ${quote(property.name)}: O.Option<${emitType(property.schema, components, decoded)}>`
                : `readonly ${quote(property.name)}?: ${emitType(property.schema, components, decoded)}`
        )
        .join("; ");
      const object = `{ ${fields} }`;
      return AdditionalProperties.$match(additionalProperties, {
        Forbidden: () => object,
        Allowed: () => `${object} & Readonly<Record<string, JsonValue>>`,
        Typed: ({ schema }) =>
          `${object} & Readonly<Record<string, ${emitType(
            schema,
            components,
            decoded
          )}>>`,
      });
    },
    Union: ({ members }) =>
      members.map((member) => `(${emitType(member, components, decoded)})`).join(" | "),
    Ref: ({ target }) =>
      `${componentName(target, components)}${decoded ? "" : "Encoded"}`,
  });

const schemaReferences = (schema: SchemaNodeType): ReadonlySet<string> => {
  const references = new Set<string>();
  const visit = (node: SchemaNodeType): void => {
    SchemaNode.$match(node, {
      JsonValue: () => undefined,
      Never: () => undefined,
      Null: () => undefined,
      Literal: () => undefined,
      Enum: () => undefined,
      String: () => undefined,
      Number: () => undefined,
      Boolean: () => undefined,
      Ref: ({ target }) => {
        references.add(target);
      },
      Array: ({ items }) => visit(items),
      Union: ({ members }) => members.forEach(visit),
      Object: ({ properties, additionalProperties }) => {
        properties.forEach(({ schema }) => visit(schema));
        AdditionalProperties.$match(additionalProperties, {
          Allowed: () => undefined,
          Forbidden: () => undefined,
          Typed: ({ schema }) => visit(schema),
        });
      },
    });
  };
  visit(schema);
  return references;
};

const recursiveComponents = (
  components: HashMap.HashMap<string, ComponentSchema>
): ReadonlySet<string> => {
  const references = new Map(
    [...HashMap.entries(components)].map(([target, component]) => [
      target,
      schemaReferences(component.schema),
    ])
  );
  const recursive = new Set<string>();
  const reaches = (
    origin: string,
    current: string,
    visited: ReadonlySet<string>
  ): boolean => {
    const dependencies = references.get(current);
    if (dependencies === undefined) return false;
    for (const dependency of dependencies) {
      if (dependency === origin) return true;
      if (
        !visited.has(dependency) &&
        reaches(origin, dependency, new Set([...visited, dependency]))
      )
        return true;
    }
    return false;
  };
  for (const target of references.keys())
    if (reaches(target, target, new Set([target]))) recursive.add(target);
  return recursive;
};

const generatedHeader = `// Generated by @magnitudedev/openapi-effect. DO NOT EDIT.\n`;

const emitSchemas = (ir: ProtocolIr): string => {
  const recursive = recursiveComponents(ir.components);
  const declarations = [...HashMap.entries(ir.components)]
    .map(([target, component]) => ({ target, component }))
    .sort((left, right) =>
      left.component.name.localeCompare(right.component.name)
    )
    .map(({ target, component }) => {
      const expression = emitSchemaExpression(component.schema, ir.components);
      if (!recursive.has(target))
        return [
          `export const ${component.name} = ${expression}`,
          `export type ${component.name} = S.Schema.Type<typeof ${component.name}>`,
          `export type ${component.name}Encoded = S.Schema.Encoded<typeof ${component.name}>`,
        ].join("\n");
      const type = emitType(component.schema, ir.components, true);
      const encodedType = emitType(component.schema, ir.components, false);
      return [
        `export type ${component.name} = ${type}`,
        `export type ${component.name}Encoded = ${encodedType}`,
        `export const ${component.name}: S.Schema<${component.name}, ${component.name}Encoded> = ${expression}`,
      ].join("\n");
    })
    .join("\n\n");
  const optionImport = declarations.includes("O.Option<")
    ? `import type * as O from "effect/Option"\n`
    : "";
  return `${generatedHeader}${optionImport}import * as S from "effect/Schema"\n\nexport type JsonValue = string | number | boolean | null | ReadonlyArray<JsonValue> | { readonly [key: string]: JsonValue }\nexport const JsonValue: S.Schema<JsonValue, JsonValue> = S.suspend((): S.Schema<JsonValue, JsonValue> => S.Union(S.String, S.Number, S.Boolean, S.Null, S.Array(JsonValue), S.Record({ key: S.String, value: JsonValue })))\n\n${declarations}\n`;
};

const moduleImport = (fromFile: string, toFile: string): string => {
  const relative = Path.posix
    .relative(Path.posix.dirname(fromFile), toFile)
    .replace(/\.ts$/, ".js");
  return relative.startsWith(".") ? relative : `./${relative}`;
};

const resolveRef = (
  schema: SchemaNodeType,
  components: HashMap.HashMap<string, ComponentSchema>,
  visited: ReadonlySet<string> = new Set()
): SchemaNodeType => {
  if (schema._tag !== "Ref" || visited.has(schema.target)) return schema;
  return HashMap.get(components, schema.target).pipe(
    Option.map(({ schema: target }) =>
      resolveRef(target, components, new Set([...visited, schema.target]))
    ),
    Option.getOrElse(() => schema)
  );
};

const emitWireSchema = (
  input: SchemaNodeType,
  components: HashMap.HashMap<string, ComponentSchema>
): string => {
  const schema = resolveRef(input, components);
  return SchemaNode.$match(schema, {
    String: (value) => withPipe("S.String", emitStringFilters(value)),
    Number: (value) => withPipe("S.NumberFromString", emitNumberFilters(value)),
    Boolean: () => "S.BooleanFromString",
    Literal: ({ value }) =>
      typeof value === "string"
        ? literalSchema(value)
        : typeof value === "number"
        ? `S.compose(S.NumberFromString, ${literalSchema(value)})`
        : typeof value === "boolean"
        ? `S.compose(S.BooleanFromString, ${literalSchema(value)})`
        : "S.Never",
    Enum: ({ values }) =>
      values.every((value) => typeof value === "string")
        ? `S.Literal(${values
            .map((value) => JSON.stringify(value))
            .join(", ")})`
        : values.every((value) => typeof value === "number")
        ? `S.compose(S.NumberFromString, S.Literal(${values.join(", ")}))`
        : values.every((value) => typeof value === "boolean")
        ? `S.compose(S.BooleanFromString, S.Literal(${values.join(", ")}))`
        : "S.Never",
    Array: ({ items }) => `S.Array(${emitWireSchema(items, components)})`,
    Union: ({ members }) =>
      `S.Union(${members
        .map((member) => emitWireSchema(member, components))
        .join(", ")})`,
    Ref: ({ target }) => `Schemas.${componentName(target, components)}`,
    JsonValue: () => "S.Never",
    Never: () => "S.Never",
    Null: () => "S.Never",
    Object: () => "S.Never",
  });
};

const emitParameterStruct = (
  parameters: readonly OperationParameter[],
  components: HashMap.HashMap<string, ComponentSchema>
): string =>
  `S.Struct({ ${parameters
    .map(
      (parameter) =>
        `${quote(parameter.name)}: ${
          parameter.required
            ? emitWireSchema(parameter.schema, components)
            : `S.optionalWith(${emitWireSchema(
                parameter.schema,
                components
              )}, { exact: true, as: "Option" })`
        }`
    )
    .join(", ")} })`;

const responseSchema = (
  response: OperationResponse,
  components: HashMap.HashMap<string, ComponentSchema>
): string =>
  Option.match(response.mediaType, {
    onNone: () => "S.Void",
    onSome: (mediaType) =>
      mediaType === "application/json"
        ? Option.match(response.schema, {
            onNone: () => "S.Never",
            onSome: (schema) =>
              emitSchemaExpression(schema, components, "Schemas."),
          })
        : mediaType === "text/plain"
        ? "HttpApiSchema.Text()"
        : "HttpApiSchema.Uint8Array()",
  });

const bodySchema = (
  mediaType: MediaType,
  schema: SchemaNodeType,
  components: HashMap.HashMap<string, ComponentSchema>
): string =>
  mediaType === "application/json"
    ? emitSchemaExpression(schema, components, "Schemas.")
    : mediaType === "text/plain"
    ? "HttpApiSchema.Text()"
    : "HttpApiSchema.Uint8Array()";

const emitEndpoint = (
  operation: Extract<OperationType, { readonly _tag: "Http" }>,
  ir: ProtocolIr
): string => {
  const method =
    operation.method === "DELETE" ? "del" : operation.method.toLowerCase();
  const path = operation.path.replaceAll(/\{([^{}]+)\}/g, ":$1");
  const lines = [
    `HttpApiEndpoint.${method}(${quote(operation.name)}, ${quote(path)})`,
  ];
  const pathParameters = operation.parameters.filter(
    ({ location }) => location === "Path"
  );
  const queryParameters = operation.parameters.filter(
    ({ location }) => location === "Query"
  );
  const headerParameters = operation.parameters.filter(
    ({ location }) => location === "Header"
  );
  if (pathParameters.length > 0)
    lines.push(
      `.setPath(${emitParameterStruct(pathParameters, ir.components)})`
    );
  if (queryParameters.length > 0)
    lines.push(
      `.setUrlParams(${emitParameterStruct(queryParameters, ir.components)})`
    );
  if (headerParameters.length > 0)
    lines.push(
      `.setHeaders(${emitParameterStruct(headerParameters, ir.components)})`
    );
  if (Option.isSome(operation.body)) {
    const expression = bodySchema(
      operation.body.value.mediaType,
      operation.body.value.schema,
      ir.components
    );
    lines.push(
      `.setPayload(${
        operation.body.value.required
          ? expression
          : `S.OptionFromUndefinedOr(${expression})`
      })`
    );
  }
  for (const response of operation.responses) {
    lines.push(
      `${response.success ? ".addSuccess" : ".addError"}(${responseSchema(
        response,
        ir.components
      )}, { status: ${response.status} })`
    );
  }
  return `export const ${operation.name} = ${lines.join("\n  ")}\n`;
};

const emitApi = (
  ir: ProtocolIr,
  config: OpenApiEffectConfig,
  schemasFile: string
): string => {
  const httpOperations = [...ir.operations].filter(Operation.$is("Http"));
  const endpoints = httpOperations
    .map((operation) => emitEndpoint(operation, ir))
    .join("\n");
  const groupNames = [
    ...new Set(httpOperations.map(({ groupName }) => groupName)),
  ].sort();
  const groups = groupNames
    .map((groupName) => {
      const operations = httpOperations.filter(
        (operation) => operation.groupName === groupName
      );
      return `export const ${pascalIdentifier(
        groupName
      )}Group = HttpApiGroup.make(${quote(groupName)})${operations
        .map(({ name }) => `\n  .add(${name})`)
        .join("")}\n`;
    })
    .join("\n");
  const apiName = pascalIdentifier(config.apiName);
  const api = `export const ${apiName} = HttpApi.make(${quote(
    config.apiName
  )})${groupNames
    .map((groupName) => `\n  .add(${pascalIdentifier(groupName)}Group)`)
    .join("")}\n`;
  return `${generatedHeader}import * as HttpApi from "@effect/platform/HttpApi"\nimport * as HttpApiEndpoint from "@effect/platform/HttpApiEndpoint"\nimport * as HttpApiGroup from "@effect/platform/HttpApiGroup"\nimport * as HttpApiSchema from "@effect/platform/HttpApiSchema"\nimport * as S from "effect/Schema"\nimport * as Schemas from ${quote(
    moduleImport(config.output.api, schemasFile)
  )}\n\n${endpoints}\n${groups}\n${api}`;
};

const pascalIdentifier = (value: string): string => {
  const words = Option.getOrElse(
    Option.fromNullable(value.match(/[A-Za-z0-9]+/g)),
    () => [] as const
  );
  const identifier =
    words
      .map((word) => `${word.charAt(0).toUpperCase()}${word.slice(1)}`)
      .join("") || "Generated";
  return /^[0-9]/.test(identifier) ? `_${identifier}` : identifier;
};

const emitOperations = (
  ir: ProtocolIr,
  config: OpenApiEffectConfig
): string => {
  const needsSchemas = [...ir.operations].some(Operation.$is("Stream"));
  const schemaImport = needsSchemas
    ? `import * as HttpApiSchema from "@effect/platform/HttpApiSchema"
import * as S from "effect/Schema"
import * as Schemas from ${quote(
        moduleImport(config.output.operations, config.output.schemas)
      )}\n\n`
    : "";
  const descriptors = [...ir.operations]
    .map((operation) =>
      Operation.$match(operation, {
        Http: (value) =>
          `export const ${value.name}Operation = { operationId: ${quote(
            value.operationId
          )}, transport: "http", method: ${quote(value.method)}, path: ${quote(
            value.path
          )}, group: ${quote(value.group)} } as const`,
        Stream: (value) => {
          const pathParameters = value.parameters.filter(
            ({ location }) => location === "Path"
          );
          const queryParameters = value.parameters.filter(
            ({ location }) => location === "Query"
          );
          const headerParameters = value.parameters.filter(
            ({ location }) => location === "Header"
          );
          const fields = [
            `operationId: ${quote(value.operationId)}`,
            `transport: ${quote(value.framing === "Sse" ? "sse" : "ndjson")}`,
            `method: ${quote(value.method)}`,
            `path: ${quote(value.path)}`,
            `group: ${quote(value.group)}`,
            `mediaType: ${quote(value.mediaType)}`,
            `responseStatus: ${value.responseStatus}`,
            `eventSchema: ${
              value.eventSchema._tag === "Ref"
                ? `Schemas.${componentName(
                    value.eventSchema.target,
                    ir.components
                  )}`
                : "Schemas.JsonValue"
            }`,
            `termination: ${
              value.termination._tag === "Sentinel"
                ? `{ type: "sentinel", value: ${quote(
                    value.termination.value
                  )} }`
                : value.termination._tag === "Eof"
                ? `{ type: "eof" }`
                : `{ type: "long-lived" }`
            }`,
            `reconnect: ${
              value.reconnect._tag === "LastEventId"
                ? `{ type: "last-event-id" }`
                : `{ type: "none" }`
            }`,
            ...(pathParameters.length > 0
              ? [
                  `pathParameters: ${emitParameterStruct(
                    pathParameters,
                    ir.components
                  )}`,
                ]
              : []),
            ...(queryParameters.length > 0
              ? [
                  `queryParameters: ${emitParameterStruct(
                    queryParameters,
                    ir.components
                  )}`,
                ]
              : []),
            ...(headerParameters.length > 0
              ? [
                  `headers: ${emitParameterStruct(
                    headerParameters,
                    ir.components
                  )}`,
                ]
              : []),
            ...Option.toArray(
              Option.map(value.body, (body) => {
                const schema = bodySchema(
                  body.mediaType,
                  body.schema,
                  ir.components
                );
                return `payload: ${
                  body.required ? schema : `S.OptionFromUndefinedOr(${schema})`
                }`;
              })
            ),
            `errors: [${value.errors
              .map(
                (response) =>
                  `{ status: ${response.status}, schema: ${responseSchema(
                    response,
                    ir.components
                  )} }`
              )
              .join(", ")}]`,
          ];
          return `export const ${value.name}Operation = { ${fields.join(
            ", "
          )} } as const`;
        },
      })
    )
    .join("\n\n");
  return `${generatedHeader}${schemaImport}${descriptors}\n`;
};

export const emitProject = (
  ir: ProtocolIr,
  config: OpenApiEffectConfig
): Effect.Effect<EmittedProject, OpenApiEmitError> => {
  const modules = [
    [config.output.schemas, emitSchemas(ir)],
    [config.output.operations, emitOperations(ir, config)],
    [config.output.api, emitApi(ir, config, config.output.schemas)],
  ] as const;
  return Effect.try({
    try: () =>
      new EmittedProject({
        files: HashMap.fromIterable(
          modules.map(
            ([path, source]) =>
              [
                path,
                format(source, {
                  parser: "typescript",
                  semi: false,
                  singleQuote: false,
                  trailingComma: "all",
                  printWidth: 120,
                }),
              ] as const
          )
        ),
      }),
    catch: (cause) =>
      new OpenApiEmitError({ module: "typescript", message: String(cause) }),
  });
};
