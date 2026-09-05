import { createHash } from "node:crypto";
import { RpcSchema, type Rpc } from "@effect/rpc";
import { Context, Effect, Option, Schema, SchemaAST as AST } from "effect";
import { canonical } from "@magnitudedev/utils/canonical-key";
import { JsonValueSchema, type JsonValue } from "@magnitudedev/utils/schema";
import {
  AcnRpcGroup,
  AcnRpcRecoveryPolicyTag,
  AcnSubscriptionWireItem,
  MagnitudeHealthResponseSchema,
} from "@magnitudedev/acn-protocol";

export class UnsupportedWireSchema extends Schema.TaggedError<UnsupportedWireSchema>()(
  "UnsupportedWireSchema",
  {
    path: Schema.String,
    construct: Schema.String,
  }
) {}

/** Encoded structural contract, not arbitrary AST JSON or decoded TypeScript identity. */
export const describeWireSchema = (
  schema: Schema.Schema.All,
  origin = "schema"
) =>
  Effect.try({
    try: () => {
      const active: AST.AST[] = [];
      const visit = (ast: AST.AST, path: string): JsonValue => {
        // These wrappers do not add wire fields. Transform behavior requires semantic review.
        if (ast._tag === "Transformation") {
          return visit(ast.from, path);
        }
        const ancestor = active.indexOf(ast);
        if (ancestor !== -1)
          return { kind: "recursive", distance: active.length - ancestor };
        active.push(ast);
        try {
          switch (ast._tag) {
            case "Suspend":
              return visit(ast.f(), path);
            case "NeverKeyword":
            case "UnknownKeyword":
            case "AnyKeyword":
            case "StringKeyword":
            case "NumberKeyword":
            case "BooleanKeyword":
            case "UndefinedKeyword":
            case "VoidKeyword":
            case "ObjectKeyword":
              return { kind: ast._tag };
            case "Literal":
              if (typeof ast.literal === "bigint")
                throw new UnsupportedWireSchema({
                  path,
                  construct: "bigint literal cannot cross JSON",
                });
              return { kind: "literal", value: ast.literal };
            case "Enums":
              return {
                kind: "enum",
                values: ast.enums
                  .map(([, value]) => value)
                  .sort((a, b) => String(a).localeCompare(String(b))),
              };
            case "TemplateLiteral":
              return {
                kind: "template",
                head: ast.head,
                spans: ast.spans.map((span) => ({
                  type: visit(span.type, path),
                  literal: span.literal,
                })),
              };
            case "TupleType":
              return {
                kind: "tuple",
                elements: ast.elements.map((element, i) => ({
                  optional: element.isOptional,
                  type: visit(element.type, `${path}[${i}]`),
                })),
                rest: ast.rest.map((element, i) =>
                  visit(element.type, `${path}[...${i}]`)
                ),
              };
            case "TypeLiteral":
              return {
                kind: "object",
                fields: [...ast.propertySignatures]
                  .sort((a, b) => String(a.name).localeCompare(String(b.name)))
                  .map((field) => {
                    if (typeof field.name === "symbol")
                      throw new UnsupportedWireSchema({
                        path,
                        construct: "symbol field",
                      });
                    return {
                      name: String(field.name),
                      optional: field.isOptional,
                      type: visit(field.type, `${path}.${String(field.name)}`),
                    };
                  }),
                indexes: ast.indexSignatures.map((index) => ({
                  key: visit(index.parameter, `${path}.[key]`),
                  value: visit(index.type, `${path}.[value]`),
                })),
              };
            // Alternative order can affect refinement/transform selection; retain it.
            case "Union":
              return {
                kind: "union",
                alternatives: ast.types.map((type, i) =>
                  visit(type, `${path}|${i}`)
                ),
              };
            case "Refinement": {
              const id = ast.annotations[AST.SchemaIdAnnotationId];
              const constraints = ast.annotations[AST.JSONSchemaAnnotationId];
              const semanticId =
                typeof id === "symbol"
                  ? Symbol.keyFor(id)
                  : typeof id === "string"
                  ? id
                  : undefined;
              return {
                kind: "refinement",
                type: visit(ast.from, path),
                ...(semanticId === undefined ? {} : { semanticId }),
                ...(constraints === undefined
                  ? {}
                  : {
                      constraints:
                        Schema.decodeUnknownSync(JsonValueSchema)(constraints),
                    }),
                ...(constraints === undefined ? { opaquePredicate: true } : {}),
              };
            }
            default:
              throw new UnsupportedWireSchema({ path, construct: ast._tag });
          }
        } finally {
          active.pop();
        }
      };
      return visit(schema.ast, origin);
    },
    catch: (error) =>
      error instanceof UnsupportedWireSchema
        ? error
        : new UnsupportedWireSchema({ path: origin, construct: String(error) }),
  });

export const rpcFingerprint = (
  group: {
    readonly requests: ReadonlyMap<string, Rpc.AnyWithProps>;
  } = AcnRpcGroup
) =>
  Effect.gen(function* () {
    const hash = (value: JsonValue): string =>
      createHash("sha256").update(canonical(value)).digest("hex");
    const describe = (schema: Schema.Schema.All, path: string) =>
      describeWireSchema(schema, path).pipe(Effect.map(hash));
    const procedures = yield* Effect.forEach(
      [...group.requests.values()].sort((a, b) => a._tag.localeCompare(b._tag)),
      (rpc) =>
        Effect.gen(function* () {
          const stream = RpcSchema.isStreamSchema(rpc.successSchema);
          const replay = stream
            ? ("Subscription" as const)
            : Option.getOrThrow(
                Context.getOption(rpc.annotations, AcnRpcRecoveryPolicyTag)
              );
          return {
            tag: rpc._tag,
            stream,
            replay,
            payload: yield* describe(rpc.payloadSchema, `${rpc._tag}.payload`),
            success: yield* describe(
              stream ? rpc.successSchema.success : rpc.successSchema,
              `${rpc._tag}.success`
            ),
            error: yield* describe(
              stream
                ? Schema.Union(rpc.errorSchema, rpc.successSchema.failure)
                : rpc.errorSchema,
              `${rpc._tag}.error`
            ),
          };
        })
    );
    return hash({
      format: 1,
      transport: {
        encoding: "effect-rpc-ndjson",
        endpoint: "/rpc",
        instanceHeader: "x-magnitude-acn-id",
        staleInstanceStatus: 409,
        unavailableStatus: 503,
      },
      health: yield* describe(MagnitudeHealthResponseSchema, "health"),
      subscription: yield* describe(AcnSubscriptionWireItem, "subscription"),
      procedures,
    });
  });
