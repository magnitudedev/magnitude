import { NdjsonTransport, OpenApiEffectConfig } from "../src/index.js";

export const config = new OpenApiEffectConfig({ apiName: "ExampleApi" });

export const ndjsonConfig = new OpenApiEffectConfig({
  apiName: "ExampleApi",
  transports: [
    new NdjsonTransport({
      extension: "x-magnitude-transport",
      value: "ndjson",
      eventSchemaExtension: "x-magnitude-event-schema",
    }),
  ],
});

export const document = {
  openapi: "3.1.0",
  info: { title: "Example", version: "1.0.0" },
  components: {
    schemas: {
      Model: {
        type: "object",
        required: ["id", "tokens"],
        properties: {
          id: { type: "string", minLength: 1 },
          tokens: { type: "integer", minimum: 0 },
          label: { type: ["string", "null"] },
        },
        additionalProperties: false,
      },
      OpenModelRequest: {
        type: "object",
        required: ["path"],
        properties: {
          path: { type: "string" },
        },
        additionalProperties: false,
      },
      Problem: {
        type: "object",
        required: ["message"],
        properties: { message: { type: "string" } },
        additionalProperties: false,
      },
      GenerationEvent: {
        oneOf: [
          {
            type: "object",
            required: ["type", "text"],
            properties: {
              type: { const: "delta" },
              text: { type: "string" },
            },
            additionalProperties: false,
          },
          {
            type: "object",
            required: ["type"],
            properties: { type: { const: "finished" } },
            additionalProperties: false,
          },
        ],
      },
    },
  },
  paths: {
    "/models/{id}": {
      get: {
        operationId: "getModel",
        tags: ["models"],
        parameters: [
          {
            name: "id",
            in: "path",
            required: true,
            schema: { type: "integer", minimum: 1 },
          },
          { name: "verbose", in: "query", schema: { type: "boolean" } },
        ],
        responses: {
          "200": {
            description: "Model",
            content: {
              "application/json": {
                schema: { $ref: "#/components/schemas/Model" },
              },
            },
          },
          "404": {
            description: "Missing",
            content: {
              "application/json": {
                schema: { $ref: "#/components/schemas/Problem" },
              },
            },
          },
        },
      },
    },
    "/models": {
      post: {
        operationId: "openModel",
        tags: ["models"],
        requestBody: {
          required: true,
          content: {
            "application/json": {
              schema: { $ref: "#/components/schemas/OpenModelRequest" },
            },
          },
        },
        responses: {
          "201": {
            description: "Opened",
            content: {
              "application/json": {
                schema: { $ref: "#/components/schemas/Model" },
              },
            },
          },
          "400": {
            description: "Invalid",
            content: {
              "application/json": {
                schema: { $ref: "#/components/schemas/Problem" },
              },
            },
          },
        },
      },
    },
  },
} as const;

export const ndjsonDocument = {
  ...document,
  paths: {
    ...document.paths,
    "/generate": {
      post: {
        operationId: "generate",
        tags: ["generation"],
        "x-magnitude-transport": "ndjson",
        "x-magnitude-event-schema": "#/components/schemas/GenerationEvent",
        requestBody: {
          required: true,
          content: {
            "application/json": {
              schema: { $ref: "#/components/schemas/OpenModelRequest" },
            },
          },
        },
        responses: {
          "200": {
            description: "Events",
            content: { "application/x-ndjson": { schema: { type: "string" } } },
          },
          "400": {
            description: "Invalid request",
            content: {
              "application/json": {
                schema: { $ref: "#/components/schemas/Problem" },
              },
            },
          },
        },
      },
    },
  },
} as const;
