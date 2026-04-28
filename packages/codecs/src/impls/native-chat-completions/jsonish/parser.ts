
/**
 * Streaming JSON Parser
 * 
 * Incremental JSON parser ported from BAML's JsonParseState.
 * Handles malformed JSON commonly produced by LLMs:
 * - Unquoted strings with smart lookahead
 * - Incomplete numbers, strings, objects, arrays
 * - Proper escape sequence handling in quoted strings
 * 
 * Skips (not needed for our use case):
 * - Markdown wrapping, multi-JSON extraction
 * - Single-quoted strings, backtick strings, triple-quoted strings
 * - Comments (// and /*)
 * - AnyOf/FixedJson multi-candidate resolution
 */

import type {
  CompletionState,
  ParsedValue,
  JsonCollection,
  CloseStringResult,
  Pos,
  StreamingJsonParser,
} from './types';

/**
 * Create a new streaming JSON parser.
 */
export function createStreamingJsonParser(): StreamingJsonParser {
  // Stack of collections being built
  const collectionStack: JsonCollection[] = [];
  
  // Completed top-level values
  const completedValues: Array<{ name: string; value: ParsedValue }> = [];
  
  // Current position in the stream (for iterator-based lookahead)
  let streamEnded = false;

  /**
   * Get the current position context based on the collection stack.
   */
  function getPos(): Pos {
    if (collectionStack.length < 2) {
      return { _tag: 'inNothing' };
    }
    
    const parent = collectionStack[collectionStack.length - 2];
    
    switch (parent._tag) {
      case 'object': {
        // In object key if keys.length === values.length (expecting key next)
        // In object value if keys.length > values.length (have key, expecting value)
        if (parent.keys.length === parent.values.length) {
          return { _tag: 'inObjectKey' };
        } else {
          return { _tag: 'inObjectValue' };
        }
      }
      case 'array':
        return { _tag: 'inArray' };
      default:
        return { _tag: 'unknown' };
    }
  }

  /**
   * Update quote tracking for O(1) escape detection.
   * Must be called BEFORE the character is added to the string.
   */
  function updateQuoteTracking(collection: { trailingBackslashes: number; unescapedQuoteCount: number }, char: string): void {
    if (char === '\\') {
      collection.trailingBackslashes += 1;
    } else {
      if (char === '"') {
        // A quote is "unescaped" if preceded by an even number of backslashes
        if (collection.trailingBackslashes % 2 === 0) {
          collection.unescapedQuoteCount += 1;
        }
      }
      collection.trailingBackslashes = 0;
    }
  }

  /**
   * Resolve an unquoted string collection to a ParsedValue.
   */
  function resolveUnquotedString(collection: { content: string; state: CompletionState }): ParsedValue {
    const trimmed = collection.content.trim();
    
    // Check for boolean literals
    if (trimmed === 'true') {
      return { _tag: 'boolean', value: true, state: 'complete' };
    }
    if (trimmed === 'false') {
      return { _tag: 'boolean', value: false, state: 'complete' };
    }
    
    // Check for null
    if (trimmed === 'null') {
      return { _tag: 'null', state: 'complete' };
    }
    
    // Check for number
    if (trimmed !== '' && !isNaN(Number(trimmed)) && isFinite(Number(trimmed))) {
      return { 
        _tag: 'number', 
        value: trimmed, 
        state: collection.state 
      };
    }
    
    // Otherwise treat as string
    return { 
      _tag: 'string', 
      value: trimmed, 
      state: collection.state 
    };
  }

  /**
   * Convert a collection to a ParsedValue.
   */
  function collectionToValue(collection: JsonCollection): ParsedValue | undefined {
    switch (collection._tag) {
      case 'object': {
        const entries: Array<[string, ParsedValue]> = [];
        const len = Math.min(collection.keys.length, collection.values.length);
        for (let i = 0; i < len; i++) {
          entries.push([collection.keys[i], collection.values[i]]);
        }
        return { 
          _tag: 'object', 
          entries, 
          state: collection.state 
        };
      }
      case 'array': {
        return { 
          _tag: 'array', 
          items: collection.items, 
          state: collection.state 
        };
      }
      case 'quotedString': {
        return { 
          _tag: 'string', 
          value: collection.content, 
          state: collection.state 
        };
      }
      case 'unquotedString': {
        return resolveUnquotedString(collection);
      }
      default:
        return undefined;
    }
  }

  /**
   * Convert any ParsedValue to a string representation.
   */
  function valueToString(value: ParsedValue): string {
    switch (value._tag) {
      case 'string': return value.value;
      case 'number': return value.value;
      case 'boolean': return String(value.value);
      case 'null': return 'null';
      case 'object': {
        const entries = value.entries.map(([k, v]) => `${k}: ${valueToString(v)}`).join(', ');
        return `{${entries}}`;
      }
      case 'array': {
        const items = value.items.map(v => valueToString(v)).join(', ');
        return `[${items}]`;
      }
    }
  }

  /**
   * Complete the top collection on the stack, merging it into parent or completedValues.
   */
  function completeCollection(completionState: CompletionState): void {
    const collection = collectionStack.pop();
    if (!collection) return;
    
    // Update the collection's state
    collection.state = completionState;
    
    const value = collectionToValue(collection);
    if (!value) return;
    
    // If there's a parent, merge into it
    if (collectionStack.length > 0) {
      const parent = collectionStack[collectionStack.length - 1];
      switch (parent._tag) {
        case 'object': {
          // Determine if we're adding a key or a value
          if (parent.keys.length === parent.values.length) {
            // Adding a key
            if (value._tag === 'string') {
              parent.keys.push(value.value);
            } else {
              // Non-string keys get stringified
              parent.keys.push(valueToString(value));
            }
          } else {
            // Adding a value
            parent.values.push(value);
          }
          break;
        }
        case 'array': {
          parent.items.push(value);
          break;
        }
      }
    } else {
      // Top-level completion
      completedValues.push({ 
        name: collection._tag === 'object' ? 'Object' : 
              collection._tag === 'array' ? 'Array' : 
              collection._tag === 'quotedString' ? 'String' : 'UnquotedString',
        value 
      });
    }
  }

  /**
   * Check if an unquoted string represents a complete JSON literal.
   */
  function isStringComplete(collection: { content: string }): boolean {
    const trimmed = collection.content.trim();
    if (trimmed === 'true' || trimmed === 'false' || trimmed === 'null') {
      return true;
    }
    // Check if it's a valid number
    return trimmed !== '' && !isNaN(Number(trimmed)) && isFinite(Number(trimmed));
  }

  /**
   * Check if a string content represents a complete JSON literal.
   */
  function isCompleteLiteral(content: string): boolean {
    const trimmed = content.trim();
    if (trimmed === 'true' || trimmed === 'false' || trimmed === 'null') {
      return true;
    }
    // Check if it's a valid number
    return trimmed !== '' && !isNaN(Number(trimmed)) && isFinite(Number(trimmed));
  }

  /**
   * Determine if the current unquoted string should close based on upcoming characters.
   * This uses smart lookahead to handle LLM-generated JSON without explicit delimiters.
   */
  function shouldCloseUnescapedString(nextChars: string): CloseStringResult {
    // Ensure nextChars is always a string
    const safeNextChars = nextChars ?? '';
    const pos = getPos();
    
    switch (pos._tag) {
      case 'inNothing': {
        // At top level, close on { or [ (start of new structure) or whitespace after complete literal
        for (let i = 0; i < safeNextChars.length; i++) {
          const c = safeNextChars[i];
          if (c === '{' || c === '[') {
            return { _tag: 'close', charsConsumed: i, completion: 'complete' };
          }
          // Check if this is whitespace after a complete literal
          if (c === ' ' || c === '\t' || c === '\n' || c === '\r') {
            const top = collectionStack[collectionStack.length - 1];
            if (top && top._tag === 'unquotedString' && isStringComplete(top)) {
              // Complete literal followed by whitespace - close it
              return { _tag: 'close', charsConsumed: i, completion: 'complete' };
            }
          }
          // Consume the character into the string
          const top = collectionStack[collectionStack.length - 1];
          if (top && top._tag === 'unquotedString') {
            top.content += c;
          }
        }
        // Exhausted without finding delimiter
        return { _tag: 'close', charsConsumed: safeNextChars.length, completion: 'incomplete' };
      }
      
      case 'inObjectKey': {
        // In object key, close on colon
        for (let i = 0; i < safeNextChars.length; i++) {
          const c = safeNextChars[i];
          if (c === ':') {
            return { _tag: 'close', charsConsumed: i, completion: 'complete' };
          }
          const top = collectionStack[collectionStack.length - 1];
          if (top && top._tag === 'unquotedString') {
            top.content += c;
          }
        }
        return { _tag: 'close', charsConsumed: safeNextChars.length, completion: 'incomplete' };
      }
      
      case 'inObjectValue': {
        // In object value, close on comma or }
        for (let i = 0; i < safeNextChars.length; i++) {
          const c = safeNextChars[i];
          if (c === ',') {
            return { _tag: 'close', charsConsumed: i, completion: 'complete' };
          }
          if (c === '}') {
            return { _tag: 'close', charsConsumed: i, completion: 'complete' };
          }
          const top = collectionStack[collectionStack.length - 1];
          if (top && top._tag === 'unquotedString') {
            top.content += c;
          }
        }
        return { _tag: 'close', charsConsumed: safeNextChars.length, completion: 'incomplete' };
      }
      
      case 'inArray': {
        // In array, close on comma or ]
        for (let i = 0; i < safeNextChars.length; i++) {
          const c = safeNextChars[i];
          if (c === ',' || c === ']') {
            return { _tag: 'close', charsConsumed: i, completion: 'complete' };
          }
          const top = collectionStack[collectionStack.length - 1];
          if (top && top._tag === 'unquotedString') {
            top.content += c;
          }
        }
        // Off-by-one fix: account for the last consumed character
        return { _tag: 'close', charsConsumed: safeNextChars.length + 1, completion: 'incomplete' };
      }
      
      case 'unknown':
      default:
        return { _tag: 'continue' };
    }
  }

  /**
   * Determine if a quoted string should close at the current position.
   * For simplicity, we close the string when we see an unescaped closing quote.
   */
  function shouldCloseString(nextChars: string, closingChar: string): boolean {
    // Get quote count for double-quoted strings to check if quote is escaped
    if (closingChar === '"') {
      const top = collectionStack[collectionStack.length - 1];
      if (top && top._tag === 'quotedString') {
        // If unescaped quote count is even, this quote closes the string
        // (each pair of unescaped quotes opens and closes)
        return top.unescapedQuoteCount % 2 === 0;
      }
    }
    
    // For other quote types, just check if we're at the end of stream
    const safeNextChars = nextChars ?? '';
    return safeNextChars.length === 0;
  }

  /**
   * Process a single character token.
   */
  function processToken(char: string, nextChars: string): number {
    const top = collectionStack[collectionStack.length - 1];
    
    if (!top) {
      // No active collection - look for a starting value
      return findAnyStartingValue(char, nextChars);
    }
    
    switch (top._tag) {
      case 'object': {
        switch (char) {
          case '}':
            completeCollection('complete');
            return 0;
          case ',':
          case ':':
            // Skip structural tokens
            return 0;
          default:
            // Look for a new key or value
            return findAnyStartingValue(char, nextChars);
        }
      }
      
      case 'array': {
        switch (char) {
          case ']':
            completeCollection('complete');
            return 0;
          case ',':
            return 0;
          default:
            return findAnyStartingValue(char, nextChars);
        }
      }
      
      case 'quotedString': {
        switch (char) {
          case '"': {
            if (shouldCloseString(nextChars, '"')) {
              completeCollection('complete');
              return 0;
            } else {
              updateQuoteTracking(top, char);
              top.content += char;
              return 0;
            }
          }
          case '\\': {
            // Handle escape sequences
            // Ensure nextChars is always a string
            const safeNextChars = nextChars ?? '';
            if (safeNextChars.length === 0) {
              // Incomplete escape at end of stream
              updateQuoteTracking(top, char);
              top.content += char;
              return 0;
            }
            const escaped = safeNextChars[0];
            switch (escaped) {
              case 'n':
                updateQuoteTracking(top, char);
                top.content += '\n';
                return 1;
              case 't':
                updateQuoteTracking(top, char);
                top.content += '\t';
                return 1;
              case 'r':
                updateQuoteTracking(top, char);
                top.content += '\r';
                return 1;
              case 'b':
                updateQuoteTracking(top, char);
                top.content += '\b';
                return 1;
              case 'f':
                updateQuoteTracking(top, char);
                top.content += '\f';
                return 1;
              case '\\':
                updateQuoteTracking(top, char);
                top.content += '\\';
                return 1;
              case '"':
                updateQuoteTracking(top, char);
                top.content += '"';
                return 1;
              case 'u': {
                // Unicode escape - consume u and next 4 hex chars
                updateQuoteTracking(top, char);
                let hex = '';
                // Safely iterate, checking bounds
                for (let i = 1; i <= 4 && i < safeNextChars.length; i++) {
                  hex += safeNextChars[i];
                }
                if (hex.length === 4) {
                  try {
                    const code = parseInt(hex, 16);
                    top.content += String.fromCharCode(code);
                    return 5; // \ + u + 4 hex chars
                  } catch {
                    // Invalid unicode, just add the raw chars
                    top.content += 'u' + hex;
                    return 1 + hex.length;
                  }
                } else {
                  // Incomplete unicode escape - add what we have
                  top.content += 'u' + hex;
                  return 1 + hex.length;
                }
              }
              default:
                // Unknown escape, consume both chars
                updateQuoteTracking(top, char);
                top.content += escaped;
                return 1;
            }
          }
          default:
            updateQuoteTracking(top, char);
            top.content += char;
            return 0;
        }
      }
      
      case 'unquotedString': {
        // Accumulate the character first
        top.content += char;
        
        // Then check if we should close
        const result = shouldCloseUnescapedString(nextChars);
        if (result._tag === 'close') {
          completeCollection(result.completion);
          return result.charsConsumed;
        }
        return 0;
      }
    }
  }

  /**
   * Attempt to start parsing a new JSON value.
   */
  function findAnyStartingValue(char: string, nextChars: string): number {
    // Ensure nextChars is always a string
    const safeNextChars = nextChars ?? '';
    
    switch (char) {
      case '{': {
        collectionStack.push({
          _tag: 'object',
          keys: [],
          values: [],
          state: 'incomplete',
        });
        return 0;
      }
      case '[': {
        collectionStack.push({
          _tag: 'array',
          items: [],
          state: 'incomplete',
        });
        return 0;
      }
      case '"': {
        collectionStack.push({
          _tag: 'quotedString',
          content: '',
          state: 'incomplete',
          trailingBackslashes: 0,
          unescapedQuoteCount: 0,
        });
        return 0;
      }
      case ' ':
      case '\t':
      case '\n':
      case '\r':
        // Skip whitespace
        return 0;
      default: {
        // Start an unquoted string
        collectionStack.push({
          _tag: 'unquotedString',
          content: char,
          state: 'incomplete',
        });
        
        // Check if we should immediately close (for complete literals)
        const result = shouldCloseUnescapedString(safeNextChars);
        if (result._tag === 'close') {
          completeCollection(result.completion);
          return result.charsConsumed;
        }
        return 0;
      }
    }
  }

  /**
   * Push a chunk of text into the parser.
   */
  function push(chunk: string): void {
    // Ensure chunk is a string
    const safeChunk = chunk ?? '';
    let i = 0;
    while (i < safeChunk.length) {
      const char = safeChunk[i];
      const nextChars = safeChunk.slice(i + 1);
      const skip = processToken(char, nextChars);
      i += 1 + skip;
    }
  }

  /**
   * Signal that the stream has ended.
   */
  function end(): void {
    streamEnded = true;
    
    // Complete any remaining collections
    while (collectionStack.length > 0) {
      const top = collectionStack[collectionStack.length - 1];
      
      // For top-level unquoted strings, check if they're complete literals
      if (collectionStack.length === 1 && top._tag === 'unquotedString') {
        if (isStringComplete(top)) {
          completeCollection('complete');
        } else {
          completeCollection('incomplete');
        }
      } else {
        // Nested collections or non-unquoted strings are incomplete
        completeCollection('incomplete');
      }
    }
  }

  /**
   * Get the current partial parse result.
   * 
   * Composes the full collection stack into a single composite ParsedValue.
   * When we're mid-parse (e.g., building an object with a key being read),
   * we need to return the outermost structure with all inner collections
   * embedded as partial values — not just the topmost collection.
   */
  function getPartial(): ParsedValue | undefined {
    // If there are completed top-level values, return the latest
    if (collectionStack.length === 0 && completedValues.length > 0) {
      return completedValues[completedValues.length - 1].value;
    }
    
    // Build the composite value from the collection stack bottom-up.
    // Each collection on the stack gets embedded into its parent.
    if (collectionStack.length === 0) return undefined;
    
    // Start by converting the bottom-most collection
    let result: ParsedValue | undefined = collectionToValue(collectionStack[0]);
    
    // Walk up the stack, embedding each child into its parent
    for (let i = 1; i < collectionStack.length; i++) {
      const child = collectionToValue(collectionStack[i]);
      const parent = collectionStack[i - 1];
      
      if (!child) continue;
      
      switch (parent._tag) {
        case 'object': {
          const entries: Array<[string, ParsedValue]> = [];
          // Add completed entries
          const len = Math.min(parent.keys.length, parent.values.length);
          for (let j = 0; j < len; j++) {
            entries.push([parent.keys[j], parent.values[j]]);
          }
          // If we have an extra key (reading value), add the child as the value
          if (parent.keys.length > parent.values.length) {
            entries.push([parent.keys[parent.keys.length - 1], child]);
          }
          result = { _tag: 'object', entries, state: parent.state };
          break;
        }
        case 'array': {
          const items = [...parent.items, child];
          result = { _tag: 'array', items, state: parent.state };
          break;
        }
        // Quoted/unquoted strings on the stack are leaf values — 
        // they get embedded into their parent above
        default:
          result = child;
          break;
      }
    }
    
    return result;
  }

  /**
   * Whether the parser has completed a top-level value.
   */
  function isDone(): boolean {
    return completedValues.length > 0;
  }

  return {
    push,
    end,
    get partial() { return getPartial(); },
    get done() { return isDone(); },
    get currentPath(): readonly string[] {
      const path: string[] = []
      for (const collection of collectionStack) {
        if (collection._tag === 'object') {
          // In object value position: keys.length > values.length → last key is current
          if (collection.keys.length > collection.values.length) {
            path.push(collection.keys[collection.keys.length - 1])
          }
        } else if (collection._tag === 'array') {
          path.push(String(collection.items.length))
        }
        // quotedString/unquotedString don't add path segments
      }
      return path
    },
  };
}
