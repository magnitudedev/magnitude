import { createHash } from "node:crypto";
import { Data, Effect, HashMap, Schema } from "effect";
import { emitProject } from "./emit.js";
import {
  OpenApiConfigDecodeError,
  OpenApiDocumentDecodeError,
  OpenApiEmitError,
  type OpenApiEffectError,
} from "./errors.js";
import { normalizeOpenApi } from "./normalize.js";
import { OpenApiEffectConfig } from "./schemas/config.js";
import { OpenApiDocument } from "./schemas/openapi.js";

export class GenerationManifest extends Schema.Class<GenerationManifest>(
  "OpenApiEffect.GenerationManifest"
)({
  generator: Schema.Literal("@magnitudedev/openapi-effect"),
  generatorVersion: Schema.String,
  protocolHash: Schema.String,
  configHash: Schema.String,
  files: Schema.Array(Schema.String),
  operations: Schema.Array(Schema.String),
}) {}

const GenerationManifestJson = Schema.parseJson(GenerationManifest, {
  space: 2,
});
const ConfigJson = Schema.parseJson(OpenApiEffectConfig);

export class GeneratedProject extends Data.Class<{
  readonly files: HashMap.HashMap<string, string>;
  readonly manifest: GenerationManifest;
}> {}

const sha256 = (value: string): string =>
  createHash("sha256").update(value).digest("hex");

export const decodeOpenApiEffectConfig = (
  input: unknown
): Effect.Effect<OpenApiEffectConfig, OpenApiConfigDecodeError> =>
  Schema.decodeUnknown(OpenApiEffectConfig)(input, {
    onExcessProperty: "error",
  }).pipe(Effect.mapError((cause) => new OpenApiConfigDecodeError({ cause })));

export const decodeOpenApiDocument = (
  input: unknown
): Effect.Effect<OpenApiDocument, OpenApiDocumentDecodeError> =>
  Schema.decodeUnknown(OpenApiDocument)(input, {
    onExcessProperty: "error",
  }).pipe(
    Effect.mapError((cause) => new OpenApiDocumentDecodeError({ cause }))
  );

export const compileOpenApi = (
  input: unknown,
  config: OpenApiEffectConfig
): Effect.Effect<GeneratedProject, OpenApiEffectError> =>
  Effect.gen(function* () {
    const document = yield* decodeOpenApiDocument(input);
    const ir = yield* normalizeOpenApi(document, config);
    const emitted = yield* emitProject(ir, config);
    const emittedEntries = [...HashMap.entries(emitted.files)].sort(
      ([left], [right]) => left.localeCompare(right)
    );
    const configJson = yield* Schema.encode(ConfigJson)(config).pipe(
      Effect.mapError(
        (cause) =>
          new OpenApiEmitError({
            module: config.output.manifest,
            message: cause.message,
          })
      )
    );
    const manifest = new GenerationManifest({
      generator: "@magnitudedev/openapi-effect",
      generatorVersion: "0.0.1",
      protocolHash: sha256(
        emittedEntries.map(([path, source]) => `${path}\0${source}`).join("\0")
      ),
      configHash: sha256(configJson),
      files: [
        ...emittedEntries.map(([path]) => path),
        config.output.manifest,
      ].sort(),
      operations: [...ir.operations]
        .map(({ operationId }) => operationId)
        .sort(),
    });
    const manifestJson = yield* Schema.encode(GenerationManifestJson)(
      manifest
    ).pipe(
      Effect.mapError(
        (cause) =>
          new OpenApiEmitError({
            module: config.output.manifest,
            message: cause.message,
          })
      )
    );
    return new GeneratedProject({
      files: HashMap.set(
        emitted.files,
        config.output.manifest,
        `${manifestJson}\n`
      ),
      manifest,
    });
  });
