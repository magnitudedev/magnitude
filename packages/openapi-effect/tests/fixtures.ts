import { OpenApiEffectConfig, StreamTransport } from "../src/index.js";

export const config = new OpenApiEffectConfig({ apiName: "ExampleApi" });

export const streamConfig = new OpenApiEffectConfig({
  apiName: "ExampleApi",
  transports: [new StreamTransport({})],
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
      LifecycleEvent: {
        oneOf: [
          {
            type: "object",
            required: ["type", "handle"],
            properties: {
              type: { const: "model_loaded" },
              handle: { type: "string" },
            },
            additionalProperties: false,
          },
          {
            type: "object",
            required: ["type", "handle"],
            properties: {
              type: { const: "model_unloaded" },
              handle: { type: "string" },
            },
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
        "x-magnitude-stream": {
          version: 1,
          responseStatus: 200,
          framing: "ndjson",
          data: {
            encoding: "json",
            schema: { $ref: "#/components/schemas/GenerationEvent" },
          },
          termination: { type: "eof" },
          reconnect: { type: "none" },
        },
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

export const sseDocument = {
  ...document,
  paths: {
    ...document.paths,
    "/chat/completions": {
      post: {
        operationId: "chatCompletions",
        tags: ["generation"],
        "x-magnitude-stream": {
          version: 1,
          responseStatus: 200,
          framing: "sse",
          data: {
            encoding: "json",
            schema: { $ref: "#/components/schemas/GenerationEvent" },
          },
          termination: { type: "sentinel", value: "[DONE]" },
          reconnect: { type: "none" },
        },
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
            description: "OpenAI-compatible event stream",
            content: { "text/event-stream": { schema: { type: "string" } } },
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
    "/events": {
      get: {
        operationId: "watchLifecycle",
        tags: ["lifecycle"],
        "x-magnitude-stream": {
          version: 1,
          responseStatus: 200,
          framing: "sse",
          data: {
            encoding: "json",
            schema: { $ref: "#/components/schemas/LifecycleEvent" },
          },
          termination: { type: "long-lived" },
          reconnect: { type: "last-event-id" },
        },
        responses: {
          "200": {
            description: "Lifecycle event stream",
            content: { "text/event-stream": { schema: { type: "string" } } },
          },
        },
      },
    },
  },
} as const;
