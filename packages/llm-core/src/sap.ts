import { BamlRuntime } from '@boundaryml/baml/native';
import { TypeBuilder } from '@boundaryml/baml/type_builder';
import { convertJsonSchemaToBaml, type JsonSchema } from './schema-converter';

export type { JsonSchema };

const BAML_SCHEMA = `generator g {
  output_type "typescript"
  output_dir "."
  version "0.214.0"
}

client Noop {
  provider anthropic
  options {
    model "claude-sonnet-4-20250514"
    api_key ""
  }
}

class Target {
  @@dynamic
}

function Parse(x: string) -> Target {
  client Noop
  prompt #""#
}

function RenderOutputFormat(x: string) -> Target {
  client Noop
  prompt #"
    {{ _.role("user") }}
    {{ x }}
    {{ ctx.output_format }}
  "#
}
`;

const runtime = BamlRuntime.fromFiles('_', { '_.baml': BAML_SCHEMA }, {});
const ctx = runtime.createContextManager();

function buildTypeBuilder(schema: JsonSchema) {
    const tb = new TypeBuilder({
        classes: new Set(['Target']),
        enums: new Set([]),
        runtime: runtime,
    });
    const targetCls = tb.classBuilder('Target', []);

    const isWrapped = !(schema.type === 'object' && schema.properties);

    if (schema.type === 'object' && schema.properties) {
        const required = schema.required || [];
        for (const [key, prop] of Object.entries(schema.properties)) {
            if (typeof prop !== 'object' || prop === null) continue;
            let ft = convertJsonSchemaToBaml(tb, prop);
            if (!required.includes(key)) ft = ft.optional();
            targetCls.addProperty(key, ft);
        }
    } else {
        targetCls.addProperty('data', convertJsonSchemaToBaml(tb, schema));
    }

    return { tb, isWrapped };
}

/**
 * Returns the BAML output format instruction string for a given JSON Schema.
 * This is what BAML normally injects via {{ ctx.output_format }} in prompts.
 */
export function outputFormatString(schema: JsonSchema): string {
    const { tb } = buildTypeBuilder(schema);
    const req = runtime.buildRequestSync('RenderOutputFormat', { x: '' }, ctx.deepClone(), tb._tb(), null, false, {});
    const body = req.body.json();
    // The rendered prompt is in messages[0].content[0].text, prefixed with "\n"
    const text: string = body.messages[0].content[0].text;
    return text.trim();
}

/**
 * Schema-aligned parser. Parses a raw LLM response string against a JSON Schema,
 * with markdown extraction, type coercion, and fuzzy matching.
 */
export function parse(raw: string, schema: JsonSchema): unknown {
    const { tb, isWrapped } = buildTypeBuilder(schema);

    const result = runtime.parseLlmResponse('Parse', raw, false, ctx.deepClone(), tb._tb(), null, {});

    if (isWrapped && result && typeof result === 'object' && 'data' in result) {
        return (result as Record<string, unknown>).data;
    }
    return result;
}
