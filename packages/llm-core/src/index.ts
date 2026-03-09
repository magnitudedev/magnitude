export * from './baml_client';
export { parse as schemaAlignedParse, outputFormatString } from './sap';
export { convertJsonSchemaToBaml, type JsonSchema } from './schema-converter';

import TypeBuilder from './baml_client/type_builder';
export { TypeBuilder };

export { configureTypeBuilder, unwrapSchemaResult } from './schema-utils';

// Re-export Collector for token usage tracking and ClientRegistry for runtime client selection
export { Collector, ClientRegistry } from '@boundaryml/baml';

// Re-export BAML error types for error handling at call sites
export {
  BamlClientHttpError,
  BamlValidationError,
  BamlTimeoutError,
  BamlClientFinishReasonError,
  BamlAbortError,
  isBamlError
} from '@boundaryml/baml';
