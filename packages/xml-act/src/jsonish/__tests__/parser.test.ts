
/**
 * Streaming JSON Parser Tests
 * 
 * Comprehensive tests for the StreamingJsonParser, including:
 * - Complete JSON parsing
 * - Incomplete JSON at every level
 * - Nested structures
 * - Escape sequences
 * - Numbers mid-stream vs complete
 * - Edge cases from BAML (off-by-one handling)
 */

import { describe, it, expect } from 'vitest';
import { createStreamingJsonParser } from '../parser';
import type { ParsedValue, CompletionState } from '../types';

/**
 * Helper to create a parser and push chunks sequentially.
 */
function createParserWithChunks(chunks: string[]) {
  const parser = createStreamingJsonParser();
  for (const chunk of chunks) {
    parser.push(chunk);
  }
  return parser;
}

/**
 * Helper to get the partial value and verify its type.
 */
function expectPartial(parser: ReturnType<typeof createStreamingJsonParser>): ParsedValue | undefined {
  return parser.partial;
}

/**
 * Helper to check completion state of a value.
 */
function expectState(value: ParsedValue | undefined, expected: CompletionState) {
  expect(value?.state).toBe(expected);
}

describe('StreamingJsonParser', () => {
  describe('Complete JSON', () => {
    it('parses empty object', () => {
      const parser = createParserWithChunks(['{}']);
      expect(parser.done).toBe(true);
      const partial = expectPartial(parser);
      expect(partial?._tag).toBe('object');
      expect(partial?.state).toBe('complete');
      if (partial?._tag === 'object') {
        expect(partial.entries).toHaveLength(0);
      }
    });

    it('parses object with single string value', () => {
      const parser = createParserWithChunks(['{"key": "value"}']);
      const partial = expectPartial(parser);
      expect(partial?._tag).toBe('object');
      expectState(partial, 'complete');
      if (partial?._tag === 'object') {
        expect(partial.entries).toHaveLength(1);
        expect(partial.entries[0][0]).toBe('key');
        expect(partial.entries[0][1]._tag).toBe('string');
        if (partial.entries[0][1]._tag === 'string') {
          expect(partial.entries[0][1].value).toBe('value');
        }
      }
    });

    it('parses object with multiple values', () => {
      const parser = createParserWithChunks(['{"a": 1, "b": 2, "c": 3}']);
      const partial = expectPartial(parser);
      expect(partial?._tag).toBe('object');
      if (partial?._tag === 'object') {
        expect(partial.entries).toHaveLength(3);
        expect(partial.entries[0][0]).toBe('a');
        expect(partial.entries[1][0]).toBe('b');
        expect(partial.entries[2][0]).toBe('c');
      }
    });

    it('parses nested objects', () => {
      const parser = createParserWithChunks(['{"outer": {"inner": "value"}}']);
      const partial = expectPartial(parser);
      expect(partial?._tag).toBe('object');
      if (partial?._tag === 'object') {
        expect(partial.entries[0][0]).toBe('outer');
        const inner = partial.entries[0][1];
        expect(inner._tag).toBe('object');
        if (inner._tag === 'object') {
          expect(inner.entries[0][0]).toBe('inner');
        }
      }
    });

    it('parses empty array', () => {
      const parser = createParserWithChunks(['[]']);
      const partial = expectPartial(parser);
      expect(partial?._tag).toBe('array');
      expectState(partial, 'complete');
      if (partial?._tag === 'array') {
        expect(partial.items).toHaveLength(0);
      }
    });

    it('parses array with values', () => {
      const parser = createParserWithChunks(['[1, 2, 3]']);
      const partial = expectPartial(parser);
      expect(partial?._tag).toBe('array');
      if (partial?._tag === 'array') {
        expect(partial.items).toHaveLength(3);
        expect(partial.items[0]._tag).toBe('number');
        expect(partial.items[1]._tag).toBe('number');
        expect(partial.items[2]._tag).toBe('number');
      }
    });

    it('parses nested arrays', () => {
      const parser = createParserWithChunks(['[[1, 2], [3, 4]]']);
      const partial = expectPartial(parser);
      expect(partial?._tag).toBe('array');
      if (partial?._tag === 'array') {
        expect(partial.items).toHaveLength(2);
        expect(partial.items[0]._tag).toBe('array');
        expect(partial.items[1]._tag).toBe('array');
      }
    });

    it('parses mixed nested structures', () => {
      const parser = createParserWithChunks(['{"arr": [1, {"nested": "val"}]}']);
      const partial = expectPartial(parser);
      expect(partial?._tag).toBe('object');
      if (partial?._tag === 'object') {
        const arr = partial.entries[0][1];
        expect(arr._tag).toBe('array');
        if (arr._tag === 'array') {
          expect(arr.items[1]._tag).toBe('object');
        }
      }
    });

    it('parses string value', () => {
      const parser = createParserWithChunks(['"hello world"']);
      const partial = expectPartial(parser);
      expect(partial?._tag).toBe('string');
      expectState(partial, 'complete');
      if (partial?._tag === 'string') {
        expect(partial.value).toBe('hello world');
      }
    });

    it('parses empty string', () => {
      const parser = createParserWithChunks(['""']);
      const partial = expectPartial(parser);
      expect(partial?._tag).toBe('string');
      if (partial?._tag === 'string') {
        expect(partial.value).toBe('');
      }
    });

    it('parses integer (incomplete until terminated)', () => {
      // A bare number at top level is incomplete — nothing terminates it
      const parser = createParserWithChunks(['42']);
      parser.end();
      const partial = expectPartial(parser);
      expect(partial?._tag).toBe('number');
      expectState(partial, 'incomplete');
      if (partial?._tag === 'number') {
        expect(partial.value).toBe('42');
      }
    });

    it('parses negative integer', () => {
      const parser = createParserWithChunks(['-123']);
      const partial = expectPartial(parser);
      expect(partial?._tag).toBe('number');
      if (partial?._tag === 'number') {
        expect(partial.value).toBe('-123');
      }
    });

    it('parses float', () => {
      const parser = createParserWithChunks(['3.14159']);
      const partial = expectPartial(parser);
      expect(partial?._tag).toBe('number');
      if (partial?._tag === 'number') {
        expect(partial.value).toBe('3.14159');
      }
    });

    it('parses scientific notation', () => {
      const parser = createParserWithChunks(['1.5e10']);
      const partial = expectPartial(parser);
      expect(partial?._tag).toBe('number');
      if (partial?._tag === 'number') {
        expect(partial.value).toBe('1.5e10');
      }
    });

    it('parses true', () => {
      const parser = createParserWithChunks(['true']);
      const partial = expectPartial(parser);
      expect(partial?._tag).toBe('boolean');
      expectState(partial, 'complete');
      if (partial?._tag === 'boolean') {
        expect(partial.value).toBe(true);
      }
    });

    it('parses false', () => {
      const parser = createParserWithChunks(['false']);
      const partial = expectPartial(parser);
      expect(partial?._tag).toBe('boolean');
      if (partial?._tag === 'boolean') {
        expect(partial.value).toBe(false);
      }
    });

    it('parses null', () => {
      const parser = createParserWithChunks(['null']);
      const partial = expectPartial(parser);
      expect(partial?._tag).toBe('null');
      expectState(partial, 'complete');
    });
  });

  describe('Incomplete JSON', () => {
    it('handles incomplete object', () => {
      const parser = createParserWithChunks(['{"key": "val']);
      parser.end();
      const partial = expectPartial(parser);
      expect(partial?._tag).toBe('object');
      expectState(partial, 'incomplete');
    });

    it('handles incomplete array', () => {
      const parser = createParserWithChunks(['[1, 2']);
      parser.end();
      const partial = expectPartial(parser);
      expect(partial?._tag).toBe('array');
      expectState(partial, 'incomplete');
    });

    it('handles incomplete string', () => {
      const parser = createParserWithChunks(['"incomplete']);
      parser.end();
      const partial = expectPartial(parser);
      expect(partial?._tag).toBe('string');
      expectState(partial, 'incomplete');
      if (partial?._tag === 'string') {
        expect(partial.value).toBe('incomplete');
      }
    });

    it('handles incomplete number', () => {
      const parser = createParserWithChunks(['12']);
      parser.end();
      const partial = expectPartial(parser);
      expect(partial?._tag).toBe('number');
      expectState(partial, 'incomplete');
      if (partial?._tag === 'number') {
        expect(partial.value).toBe('12');
      }
    });

    it('handles empty input', () => {
      const parser = createStreamingJsonParser();
      parser.end();
      expect(parser.partial).toBeUndefined();
    });

    it('handles whitespace only', () => {
      const parser = createParserWithChunks(['   \n\t  ']);
      parser.end();
      expect(parser.partial).toBeUndefined();
    });
  });

  describe('Streaming / Chunked Parsing', () => {
    it('builds object incrementally', () => {
      const parser = createStreamingJsonParser();
      
      parser.push('{"');
      let partial = parser.partial;
      expect(partial?._tag).toBe('object');
      
      parser.push('key');
      partial = parser.partial;
      expect(partial?._tag).toBe('object');
      
      parser.push('": "');
      partial = parser.partial;
      expect(partial?._tag).toBe('object');
      
      parser.push('value');
      partial = parser.partial;
      expect(partial?._tag).toBe('object');
      
      parser.push('"}');
      partial = parser.partial;
      expect(partial?._tag).toBe('object');
      expectState(partial, 'complete');
    });

    it('builds array incrementally', () => {
      const parser = createStreamingJsonParser();
      
      parser.push('[1');
      let partial = parser.partial;
      expect(partial?._tag).toBe('array');
      if (partial?._tag === 'array') {
        expect(partial.items).toHaveLength(1);
      }
      
      parser.push(', 2, 3]');
      partial = parser.partial;
      expect(partial?._tag).toBe('array');
      if (partial?._tag === 'array') {
        expect(partial.items).toHaveLength(3);
      }
      expectState(partial, 'complete');
    });

    it('handles single character chunks', () => {
      const chunks = '{"a": 1}'.split('');
      const parser = createParserWithChunks(chunks);
      const partial = expectPartial(parser);
      expect(partial?._tag).toBe('object');
      expectState(partial, 'complete');
    });

    it('handles arbitrary chunk boundaries', () => {
      // Split at awkward places
      const chunks = ['{"a', '":', ' "val', 'ue', '"}'];
      const parser = createParserWithChunks(chunks);
      const partial = expectPartial(parser);
      expect(partial?._tag).toBe('object');
      expectState(partial, 'complete');
    });
  });

  describe('Escape Sequences', () => {
    it('handles newline escape', () => {
      const parser = createParserWithChunks(['"line1\\nline2"']);
      const partial = expectPartial(parser);
      expect(partial?._tag).toBe('string');
      if (partial?._tag === 'string') {
        expect(partial.value).toBe('line1\nline2');
      }
    });

    it('handles tab escape', () => {
      const parser = createParserWithChunks(['"col1\\tcol2"']);
      const partial = expectPartial(parser);
      if (partial?._tag === 'string') {
        expect(partial.value).toBe('col1\tcol2');
      }
    });

    it('handles carriage return escape', () => {
      const parser = createParserWithChunks(['"text\\r"']);
      const partial = expectPartial(parser);
      if (partial?._tag === 'string') {
        expect(partial.value).toBe('text\r');
      }
    });

    it('handles backspace escape', () => {
      const parser = createParserWithChunks(['"text\\b"']);
      const partial = expectPartial(parser);
      if (partial?._tag === 'string') {
        expect(partial.value).toBe('text\b');
      }
    });

    it('handles form feed escape', () => {
      const parser = createParserWithChunks(['"text\\f"']);
      const partial = expectPartial(parser);
      if (partial?._tag === 'string') {
        expect(partial.value).toBe('text\f');
      }
    });

    it('handles backslash escape', () => {
      const parser = createParserWithChunks(['"path\\\\to\\\\file"']);
      const partial = expectPartial(parser);
      if (partial?._tag === 'string') {
        expect(partial.value).toBe('path\\to\\file');
      }
    });

    it('handles quote escape', () => {
      const parser = createParserWithChunks(['"say \\"hello\\""']);
      const partial = expectPartial(parser);
      if (partial?._tag === 'string') {
        expect(partial.value).toBe('say "hello"');
      }
    });

    it('handles unicode escape', () => {
      const parser = createParserWithChunks(['"\\u0048\\u0065\\u006c\\u006c\\u006f"']);
      const partial = expectPartial(parser);
      if (partial?._tag === 'string') {
        expect(partial.value).toBe('Hello');
      }
    });

    it('handles unicode emoji', () => {
      const parser = createParserWithChunks(['"\\u2764"']);
      const partial = expectPartial(parser);
      if (partial?._tag === 'string') {
        expect(partial.value).toBe('❤');
      }
    });

    it('handles multiple escapes', () => {
      const parser = createParserWithChunks(['"tab\\there\\nnewline\\r\\n\\"quoted\\""']);
      const partial = expectPartial(parser);
      if (partial?._tag === 'string') {
        expect(partial.value).toBe('tab\there\nnewline\r\n"quoted"');
      }
    });

    it('handles escaped quote at chunk boundary', () => {
      const parser = createStreamingJsonParser();
      parser.push('"say \\"');
      parser.push('hello\\""');
      const partial = parser.partial;
      expect(partial?._tag).toBe('string');
      if (partial?._tag === 'string') {
        expect(partial.value).toBe('say "hello"');
      }
    });
  });

  describe('Unquoted Values', () => {
    it('parses unquoted true', () => {
      const parser = createParserWithChunks(['{key: true}']);
      const partial = expectPartial(parser);
      expect(partial?._tag).toBe('object');
      if (partial?._tag === 'object') {
        const val = partial.entries[0][1];
        expect(val._tag).toBe('boolean');
        if (val._tag === 'boolean') {
          expect(val.value).toBe(true);
        }
      }
    });

    it('parses unquoted false', () => {
      const parser = createParserWithChunks(['{key: false}']);
      const partial = expectPartial(parser);
      if (partial?._tag === 'object') {
        const val = partial.entries[0][1];
        expect(val._tag).toBe('boolean');
      }
    });

    it('parses unquoted null', () => {
      const parser = createParserWithChunks(['{key: null}']);
      const partial = expectPartial(parser);
      if (partial?._tag === 'object') {
        const val = partial.entries[0][1];
        expect(val._tag).toBe('null');
      }
    });

    it('parses unquoted number', () => {
      const parser = createParserWithChunks(['{key: 42}']);
      const partial = expectPartial(parser);
      if (partial?._tag === 'object') {
        const val = partial.entries[0][1];
        expect(val._tag).toBe('number');
        if (val._tag === 'number') {
          expect(val.value).toBe('42');
        }
      }
    });

    it('parses unquoted string', () => {
      const parser = createParserWithChunks(['{key: hello}']);
      const partial = expectPartial(parser);
      if (partial?._tag === 'object') {
        const val = partial.entries[0][1];
        expect(val._tag).toBe('string');
        if (val._tag === 'string') {
          expect(val.value).toBe('hello');
        }
      }
    });
  });

  describe('Numbers', () => {
    it('distinguishes incomplete from complete numbers', () => {
      // Incomplete - just "1"
      const parser1 = createStreamingJsonParser();
      parser1.push('1');
      parser1.end();
      const partial1 = parser1.partial;
      expect(partial1?._tag).toBe('number');
      expectState(partial1, 'incomplete');

      // Complete - "1" followed by comma in array
      const parser2 = createParserWithChunks(['[1, 2]']);
      const partial2 = expectPartial(parser2);
      expect(partial2?._tag).toBe('array');
      if (partial2?._tag === 'array') {
        expect(partial2.items[0]._tag).toBe('number');
        expect(partial2.items[0].state).toBe('complete');
      }
    });

    it('handles zero', () => {
      const parser = createParserWithChunks(['0']);
      const partial = expectPartial(parser);
      expect(partial?._tag).toBe('number');
      if (partial?._tag === 'number') {
        expect(partial.value).toBe('0');
      }
    });

    it('handles negative zero', () => {
      const parser = createParserWithChunks(['-0']);
      const partial = expectPartial(parser);
      expect(partial?._tag).toBe('number');
      if (partial?._tag === 'number') {
        expect(partial.value).toBe('-0');
      }
    });

    it('handles very large numbers', () => {
      const parser = createParserWithChunks(['9007199254740991']);
      const partial = expectPartial(parser);
      expect(partial?._tag).toBe('number');
    });

    it('handles very small decimals', () => {
      const parser = createParserWithChunks(['0.0000001']);
      const partial = expectPartial(parser);
      expect(partial?._tag).toBe('number');
      if (partial?._tag === 'number') {
        expect(partial.value).toBe('0.0000001');
      }
    });
  });

  describe('Edge Cases', () => {
    it('handles deeply nested structures', () => {
      const deep = '{"a":{"b":{"c":{"d":{"e":1}}}}}';
      const parser = createParserWithChunks([deep]);
      const partial = expectPartial(parser);
      expect(partial?._tag).toBe('object');
      expectState(partial, 'complete');
    });

    it('handles empty string in object', () => {
      const parser = createParserWithChunks(['{"key": ""}']);
      const partial = expectPartial(parser);
      if (partial?._tag === 'object') {
        const val = partial.entries[0][1];
        expect(val._tag).toBe('string');
        if (val._tag === 'string') {
          expect(val.value).toBe('');
        }
      }
    });

    it('handles string with only whitespace', () => {
      const parser = createParserWithChunks(['"   "']);
      const partial = expectPartial(parser);
      if (partial?._tag === 'string') {
        expect(partial.value).toBe('   ');
      }
    });

    it('handles consecutive special characters in string', () => {
      const parser = createParserWithChunks(['"\\\\\\n\\t\\""']);
      const partial = expectPartial(parser);
      if (partial?._tag === 'string') {
        expect(partial.value).toBe('\\\n\t"');
      }
    });

    it('handles object with many keys', () => {
      const obj = '{"a":1,"b":2,"c":3,"d":4,"e":5,"f":6,"g":7,"h":8,"i":9,"j":10}';
      const parser = createParserWithChunks([obj]);
      const partial = expectPartial(parser);
      if (partial?._tag === 'object') {
        expect(partial.entries).toHaveLength(10);
      }
    });

    it('handles array with many items', () => {
      const arr = '[1,2,3,4,5,6,7,8,9,10]';
      const parser = createParserWithChunks([arr]);
      const partial = expectPartial(parser);
      if (partial?._tag === 'array') {
        expect(partial.items).toHaveLength(10);
      }
    });

    it('handles string with unicode surrogate pair', () => {
      // Emoji represented as surrogate pair
      const parser = createParserWithChunks(['"\\ud83d\\ude00"']);
      const partial = expectPartial(parser);
      if (partial?._tag === 'string') {
        expect(partial.value).toBe('😀');
      }
    });
  });

  describe('BAML Off-By-One Edge Cases', () => {
    it('handles exhausted iterator in InArray context', () => {
      // This tests the counter += 1 fix for the off-by-one bug
      const parser = createStreamingJsonParser();
      parser.push('[');
      parser.push('hello'); // Unquoted string in array
      parser.end();
      
      const partial = parser.partial;
      expect(partial?._tag).toBe('array');
      if (partial?._tag === 'array') {
        expect(partial.items).toHaveLength(1);
        expect(partial.items[0]._tag).toBe('string');
      }
    });

    it('handles exhausted iterator in InObjectKey context', () => {
      const parser = createStreamingJsonParser();
      parser.push('{');
      parser.push('key'); // Unquoted key
      parser.end();
      
      const partial = parser.partial;
      expect(partial?._tag).toBe('object');
      expectState(partial, 'incomplete');
    });

    it('handles exhausted iterator in InObjectValue context', () => {
      const parser = createStreamingJsonParser();
      parser.push('{"key": ');
      parser.push('value'); // Unquoted value
      parser.end();
      
      const partial = parser.partial;
      expect(partial?._tag).toBe('object');
      expectState(partial, 'incomplete');
      if (partial?._tag === 'object') {
        expect(partial.entries[0][0]).toBe('key');
      }
    });

    it('handles exhausted iterator in InNothing context', () => {
      const parser = createStreamingJsonParser();
      parser.push('hello'); // Top-level unquoted
      parser.end();
      
      const partial = parser.partial;
      expect(partial?._tag).toBe('string');
      expect(partial?.state).toBe('incomplete');
    });
  });

  describe('Complex Streaming Scenarios', () => {
    it('handles object streaming with partial keys', () => {
      const parser = createStreamingJsonParser();
      
      parser.push('{"na');
      let partial = parser.partial;
      expect(partial?._tag).toBe('object');
      
      parser.push('me": "');
      partial = parser.partial;
      expect(partial?._tag).toBe('object');
      
      parser.push('John');
      partial = parser.partial;
      expect(partial?._tag).toBe('object');
      
      parser.push('"}');
      partial = parser.partial;
      expectState(partial, 'complete');
      if (partial?._tag === 'object') {
        expect(partial.entries[0][0]).toBe('name');
      }
    });

    it('handles multiple top-level values (takes last)', () => {
      const parser = createParserWithChunks(['1 2 3']);
      const partial = expectPartial(parser);
      // Should have the last complete value
      expect(partial?._tag).toBe('number');
      if (partial?._tag === 'number') {
        expect(partial.value).toBe('3');
      }
    });

    it('handles whitespace between tokens', () => {
      const parser = createParserWithChunks(['  {  "key"  :  "value"  }  ']);
      const partial = expectPartial(parser);
      expect(partial?._tag).toBe('object');
      expectState(partial, 'complete');
    });

    it('handles newlines in JSON', () => {
      const parser = createParserWithChunks(['{\n  "key": "value"\n}']);
      const partial = expectPartial(parser);
      expect(partial?._tag).toBe('object');
      expectState(partial, 'complete');
    });
  });
});
