import {
  OpenApiEffectConfig,
  OutputLayout,
  StreamTransport,
} from "@magnitudedev/openapi-effect";

export const config = new OpenApiEffectConfig({
  apiName: "IcnApi",
  output: new OutputLayout({
    schemas: "schemas.ts",
    operations: "operations.ts",
    api: "api.ts",
    client: "client.ts",
    index: "index.ts",
    manifest: "manifest.json",
  }),
  transports: [new StreamTransport({ extension: "x-magnitude-stream" })],
});
