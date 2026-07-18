import { Effect, HashMap, Option } from "effect";
import { describe, expect, it } from "vitest";
import {
  compileOpenApi,
  decodeOpenApiDocument,
  OpenApiDocumentDecodeError,
  OpenApiSemanticError,
} from "../src/index.js";
import {
  config,
  document,
  ndjsonDocument,
  sseDocument,
  streamConfig,
} from "./fixtures.js";

const compile = (input: unknown) =>
  Effect.runPromise(compileOpenApi(input, config));

describe("compileOpenApi", () => {
  it("decodes optional OpenAPI fields into Option at the wire boundary", async () => {
    const decoded = await Effect.runPromise(decodeOpenApiDocument(document));

    expect(Option.isSome(decoded.components)).toBe(true);
    expect(Option.isSome(decoded.paths)).toBe(true);
    expect(Option.isNone(decoded.webhooks)).toBe(true);
    expect(Option.isNone(decoded.servers)).toBe(true);
  });

  it("decodes OpenAPI once and emits Schema, operation, HttpApi, and manifest modules", async () => {
    const result = await compile(document);
    const schemas = HashMap.get(result.files, "schemas.ts");
    const api = HashMap.get(result.files, "api.ts");
    const manifest = HashMap.get(result.files, "manifest.json");

    expect(schemas._tag).toBe("Some");
    expect(
      schemas.pipe((value) => (value._tag === "Some" ? value.value : ""))
    ).toContain("export const Model");
    expect(
      schemas.pipe((value) => (value._tag === "Some" ? value.value : ""))
    ).toContain("export type Model = S.Schema.Type<typeof Model>");
    expect(
      schemas.pipe((value) => (value._tag === "Some" ? value.value : ""))
    ).toContain("export type ModelEncoded = S.Schema.Encoded<typeof Model>");
    expect(
      schemas.pipe((value) => (value._tag === "Some" ? value.value : ""))
    ).toContain('as: "Option"');
    expect(
      api.pipe((value) => (value._tag === "Some" ? value.value : ""))
    ).toContain("HttpApiEndpoint.get");
    expect(
      api.pipe((value) => (value._tag === "Some" ? value.value : ""))
    ).toContain("S.NumberFromString");
    expect(manifest._tag).toBe("Some");
    expect(result.manifest.operations).toEqual(["getModel", "openModel"]);
  });

  it("emits configured NDJSON operations as descriptors, not HttpApi endpoints", async () => {
    const result = await Effect.runPromise(
      compileOpenApi(ndjsonDocument, streamConfig)
    );
    const operations = HashMap.get(result.files, "operations.ts");
    const api = HashMap.get(result.files, "api.ts");
    const operationSource = operations._tag === "Some" ? operations.value : "";
    const apiSource = api._tag === "Some" ? api.value : "";

    expect(operationSource).toContain('transport: "ndjson"');
    expect(operationSource).toContain("eventSchema: Schemas.GenerationEvent");
    expect(operationSource).toContain("payload:");
    expect(operationSource).toContain("Schemas.OpenModelRequest");
    expect(operationSource).toContain("status: 400");
    expect(operationSource).toContain("Schemas.Problem");
    expect(apiSource).not.toContain('HttpApiEndpoint.post("generate"');
  });

  it("emits finite and long-lived SSE descriptors with explicit policies", async () => {
    const result = await Effect.runPromise(
      compileOpenApi(sseDocument, streamConfig)
    );
    const operations = HashMap.get(result.files, "operations.ts");
    const api = HashMap.get(result.files, "api.ts");
    const operationSource = operations._tag === "Some" ? operations.value : "";
    const apiSource = api._tag === "Some" ? api.value : "";

    expect(operationSource).toContain('transport: "sse"');
    expect(operationSource).toContain('value: "[DONE]"');
    expect(operationSource).toContain('type: "long-lived"');
    expect(operationSource).toContain('type: "last-event-id"');
    expect(operationSource).toContain("eventSchema: Schemas.LifecycleEvent");
    expect(apiSource).not.toContain('HttpApiEndpoint.post("chatCompletions"');
    expect(apiSource).not.toContain('HttpApiEndpoint.get("watchLifecycle"');
  });

  it("rejects document-shape failures through a typed decode error", async () => {
    const exit = await Effect.runPromiseExit(
      decodeOpenApiDocument({
        openapi: "3.0.0",
        info: { title: "Bad", version: "1" },
      })
    );
    expect(exit._tag).toBe("Failure");
    if (exit._tag === "Failure" && exit.cause._tag === "Fail") {
      expect(exit.cause.error).toBeInstanceOf(OpenApiDocumentDecodeError);
    }
  });

  it("rejects unsupported semantics through a typed semantic error", async () => {
    const overlapping = {
      openapi: "3.1.0",
      info: { title: "Bad", version: "1" },
      components: {
        schemas: {
          Bad: {
            oneOf: [{ type: "string" }, { type: "string", minLength: 1 }],
          },
        },
      },
      paths: {},
    };
    const exit = await Effect.runPromiseExit(
      compileOpenApi(overlapping, config)
    );
    expect(exit._tag).toBe("Failure");
    if (exit._tag === "Failure" && exit.cause._tag === "Fail") {
      expect(exit.cause.error).toBeInstanceOf(OpenApiSemanticError);
      if (exit.cause.error instanceof OpenApiSemanticError) {
        expect(exit.cause.error.diagnostics.map(({ code }) => code)).toContain(
          "schema.oneOf-overlap"
        );
      }
    }
  });

  it("is deterministic across input record ordering", async () => {
    const reversed = {
      ...document,
      components: {
        schemas: Object.fromEntries(
          Object.entries(document.components.schemas).reverse()
        ),
      },
      paths: Object.fromEntries(Object.entries(document.paths).reverse()),
    };
    const [left, right] = await Promise.all([
      compile(document),
      compile(reversed),
    ]);
    expect([...HashMap.entries(left.files)].sort()).toEqual(
      [...HashMap.entries(right.files)].sort()
    );
  });
});
