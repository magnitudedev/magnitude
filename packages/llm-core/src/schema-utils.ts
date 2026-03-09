import { TypeBuilder } from '@boundaryml/baml/type_builder';
import { convertJsonSchemaToBaml } from './schema-converter';

/**
 * Configures a TypeBuilder with schema properties for BAML extraction.
 * Returns whether the schema was wrapped (for later unwrapping).
 */
export function configureTypeBuilder(
    tb: TypeBuilder,
    className: 'ExtractedData' | 'GeminiExtractedData' | 'OpenAIExtractedData' | 'WebSearchData',
    schema: Record<string, unknown>
): boolean {
    const isWrappedSchema = !(schema.type === 'object' && schema.properties);
    const classBuilder = (tb as any)[className];

    if (schema.type === 'object' && schema.properties) {
        for (const [key, property] of Object.entries(schema.properties as Record<string, unknown>)) {
            classBuilder.addProperty(key, convertJsonSchemaToBaml(tb, property as Record<string, unknown>));
        }
    } else {
        classBuilder.addProperty('data', convertJsonSchemaToBaml(tb, schema));
    }

    return isWrappedSchema;
}

/**
 * Unwraps schema results if they were wrapped during type building.
 */
export function unwrapSchemaResult(result: unknown, wasWrapped: boolean): unknown {
    if (wasWrapped && result && typeof result === 'object' && 'data' in result) {
        return (result as Record<string, unknown>).data;
    }
    return result;
}
