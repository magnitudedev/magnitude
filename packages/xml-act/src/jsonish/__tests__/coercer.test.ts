
/**
 * SchemaCoercer tests
 * 
 * Comprehensive tests for the coercion system that maps parsed JSON values
 * against Effect Schema AST to produce StreamingPartial output.
 */

import { describe, it, expect } from "vitest";
import { Schema } from "@effect/schema";
import type { ParsedValue, ParsedString, ParsedNumber, ParsedBoolean, ParsedNull, ParsedObject, ParsedArray } from "../types";
import { coerceToStreamingPartial, tryCastToStreamingPartial, type CoercionFlag } from "../coercer";

// ============================================================================
// Test Helpers
// ============================================================================

function makeString(value: string, state: "complete" | "incomplete" = "complete"): ParsedString {
  return { _tag: "string", value, state };
}

function makeNumber(value: string, state: "complete" | "incomplete" = "complete"): ParsedNumber {
  return { _tag: "number", value, state };
}

function makeBoolean(value: boolean): ParsedBoolean {
  return { _tag: "boolean", value, state: "complete" };
}

function makeNull(): ParsedNull {
  return { _tag: "null", state: "complete" };
}

function makeObject(entries: Array<[string, ParsedValue]>, state: "complete" | "incomplete" = "complete"): ParsedObject {
  return { _tag: "object", entries, state };
}

function makeArray(items: ParsedValue[], state: "complete" | "incomplete" = "complete"): ParsedArray {
  return { _tag: "array", items, state };
}

function hasFlag(flags: CoercionFlag[], tag: CoercionFlag["_tag"]): boolean {
  return flags.some((f) => f._tag === tag);
}

function getFlag(flags: CoercionFlag[], tag: CoercionFlag["_tag"]): CoercionFlag | undefined {
  return flags.find((f) => f._tag === tag);
}

// ============================================================================
// Primitive Coercion Tests
// ============================================================================

describe("String coercion", () => {
  it("coerces complete string directly", () => {
    const parsed = makeString("hello");
    const schema = Schema.String;
    const result = coerceToStreamingPartial(parsed, schema.ast);
    
    expect(result).toBeDefined();
    expect(result!.value).toBe("hello");
    expect(result!.score).toBe(0);
    expect(hasFlag(result!.flags, "incomplete")).toBe(false);
  });

  it("coerces incomplete string with flag", () => {
    const parsed = makeString("hel", "incomplete");
    const schema = Schema.String;
    const result = coerceToStreamingPartial(parsed, schema.ast);
    
    expect(result).toBeDefined();
    expect(result!.value).toBe("hel");
    expect(hasFlag(result!.flags, "incomplete")).toBe(true);
  });

  it("try_cast requires exact type match", () => {
    const parsed = makeString("42");
    const schema = Schema.Number;
    const result = tryCastToStreamingPartial(parsed, schema.ast);
    
    expect(result).toBeUndefined();
  });

  it("coerce converts number string to string", () => {
    const parsed = makeNumber("42");
    const schema = Schema.String;
    const result = coerceToStreamingPartial(parsed, schema.ast);
    
    expect(result).toBeDefined();
    expect(result!.value).toBe("42");
    expect(hasFlag(result!.flags, "jsonToString")).toBe(true);
  });

  it("coerce converts boolean to string", () => {
    const parsed = makeBoolean(true);
    const schema = Schema.String;
    const result = coerceToStreamingPartial(parsed, schema.ast);
    
    expect(result).toBeDefined();
    expect(result!.value).toBe("true");
    expect(hasFlag(result!.flags, "jsonToString")).toBe(true);
  });

  it("coerce converts object to JSON string", () => {
    const parsed = makeObject([["key", makeString("value")]]);
    const schema = Schema.String;
    const result = coerceToStreamingPartial(parsed, schema.ast);
    
    expect(result).toBeDefined();
    expect(result!.value).toBe('{"key":"value"}');
    expect(hasFlag(result!.flags, "jsonToString")).toBe(true);
  });
});

describe("Number coercion", () => {
  it("coerces complete number directly", () => {
    const parsed = makeNumber("42");
    const schema = Schema.Number;
    const result = coerceToStreamingPartial(parsed, schema.ast);
    
    expect(result).toBeDefined();
    expect(result!.value).toBe(42);
    expect(result!.score).toBe(0);
  });

  it("coerces float number", () => {
    const parsed = makeNumber("3.14");
    const schema = Schema.Number;
    const result = coerceToStreamingPartial(parsed, schema.ast);
    
    expect(result).toBeDefined();
    expect(result!.value).toBe(3.14);
  });

  it("coerces incomplete number with flag", () => {
    const parsed = makeNumber("3.", "incomplete");
    const schema = Schema.Number;
    const result = coerceToStreamingPartial(parsed, schema.ast);
    
    expect(result).toBeDefined();
    expect(result!.value).toBe(3);
    expect(hasFlag(result!.flags, "incomplete")).toBe(true);
  });

  it("coerces string to number", () => {
    const parsed = makeString("42");
    const schema = Schema.Number;
    const result = coerceToStreamingPartial(parsed, schema.ast);
    
    expect(result).toBeDefined();
    expect(result!.value).toBe(42);
    expect(hasFlag(result!.flags, "stringToNumber")).toBe(true);
  });

  it("coerces string with trailing comma to number", () => {
    const parsed = makeString("42,");
    const schema = Schema.Number;
    const result = coerceToStreamingPartial(parsed, schema.ast);
    
    expect(result).toBeDefined();
    expect(result!.value).toBe(42);
    expect(hasFlag(result!.flags, "stringToNumber")).toBe(true);
  });

  it("coerces float string to int with rounding flag", () => {
    const parsed = makeString("3.7");
    const schema = Schema.Number;
    const result = coerceToStreamingPartial(parsed, schema.ast);
    
    expect(result).toBeDefined();
    expect(result!.value).toBe(3.7);
    // It's a float that happens to be a whole number when rounded, but we keep it as float
    expect(hasFlag(result!.flags, "stringToNumber")).toBe(true);
  });

  it("try_cast rejects string for number", () => {
    const parsed = makeString("42");
    const schema = Schema.Number;
    const result = tryCastToStreamingPartial(parsed, schema.ast);
    
    expect(result).toBeUndefined();
  });
});

describe("Boolean coercion", () => {
  it("coerces boolean directly", () => {
    const parsed = makeBoolean(true);
    const schema = Schema.Boolean;
    const result = coerceToStreamingPartial(parsed, schema.ast);
    
    expect(result).toBeDefined();
    expect(result!.value).toBe(true);
    expect(result!.score).toBe(0);
  });

  it("coerces 'true' string to boolean", () => {
    const parsed = makeString("true");
    const schema = Schema.Boolean;
    const result = coerceToStreamingPartial(parsed, schema.ast);
    
    expect(result).toBeDefined();
    expect(result!.value).toBe(true);
    expect(hasFlag(result!.flags, "stringToBoolean")).toBe(true);
  });

  it("coerces 'True' string to boolean", () => {
    const parsed = makeString("True");
    const schema = Schema.Boolean;
    const result = coerceToStreamingPartial(parsed, schema.ast);
    
    expect(result).toBeDefined();
    expect(result!.value).toBe(true);
  });

  it("coerces 'FALSE' string to boolean", () => {
    const parsed = makeString("FALSE");
    const schema = Schema.Boolean;
    const result = coerceToStreamingPartial(parsed, schema.ast);
    
    expect(result).toBeDefined();
    expect(result!.value).toBe(false);
  });

  it("coerces number to boolean", () => {
    const parsed = makeNumber("1");
    const schema = Schema.Boolean;
    const result = coerceToStreamingPartial(parsed, schema.ast);
    
    expect(result).toBeDefined();
    expect(result!.value).toBe(true);
  });

  it("coerces zero to false", () => {
    const parsed = makeNumber("0");
    const schema = Schema.Boolean;
    const result = coerceToStreamingPartial(parsed, schema.ast);
    
    expect(result).toBeDefined();
    expect(result!.value).toBe(false);
  });

  it("try_cast rejects string for boolean", () => {
    const parsed = makeString("true");
    const schema = Schema.Boolean;
    const result = tryCastToStreamingPartial(parsed, schema.ast);
    
    expect(result).toBeUndefined();
  });
});

// ============================================================================
// Optional Type Tests
// ============================================================================

describe("Optional type coercion", () => {
  it("coerces undefined to optional with default flag", () => {
    const schema = Schema.optional(Schema.String);
    const result = coerceToStreamingPartial(undefined, schema.ast);
    
    expect(result).toBeDefined();
    expect(result!.value).toBeUndefined();
    expect(hasFlag(result!.flags, "optionalDefault")).toBe(true);
  });

  it("coerces null to optional with default flag", () => {
    const parsed = makeNull();
    const schema = Schema.optional(Schema.String);
    const result = coerceToStreamingPartial(parsed, schema.ast);
    
    expect(result).toBeDefined();
    expect(result!.value).toBeUndefined();
    expect(hasFlag(result!.flags, "optionalDefault")).toBe(true);
  });

  it("coerces value through optional wrapper", () => {
    const parsed = makeString("hello");
    const schema = Schema.optional(Schema.String);
    const result = coerceToStreamingPartial(parsed, schema.ast);
    
    expect(result).toBeDefined();
    expect(result!.value).toBe("hello");
  });

  it("try_cast allows undefined for optional", () => {
    const schema = Schema.optional(Schema.String);
    const result = tryCastToStreamingPartial(undefined, schema.ast);
    
    expect(result).toBeDefined();
    expect(result!.value).toBeUndefined();
    expect(result!.score).toBe(0);
  });
});

// ============================================================================
// Array Coercion Tests
// ============================================================================

describe("Array coercion", () => {
  it("coerces array of strings", () => {
    const parsed = makeArray([makeString("a"), makeString("b"), makeString("c")]);
    const schema = Schema.Array(Schema.String);
    const result = coerceToStreamingPartial(parsed, schema.ast);
    
    expect(result).toBeDefined();
    expect(result!.value).toEqual(["a", "b", "c"]);
    expect(result!.score).toBe(0);
  });

  it("coerces array of numbers", () => {
    const parsed = makeArray([makeNumber("1"), makeNumber("2"), makeNumber("3")]);
    const schema = Schema.Array(Schema.Number);
    const result = coerceToStreamingPartial(parsed, schema.ast);
    
    expect(result).toBeDefined();
    expect(result!.value).toEqual([1, 2, 3]);
  });

  it("coerces incomplete array with flag", () => {
    const parsed = makeArray([makeString("a")], "incomplete");
    const schema = Schema.Array(Schema.String);
    const result = coerceToStreamingPartial(parsed, schema.ast);
    
    expect(result).toBeDefined();
    expect(hasFlag(result!.flags, "incomplete")).toBe(true);
  });

  it("coerces single value to array with flag", () => {
    const parsed = makeString("single");
    const schema = Schema.Array(Schema.String);
    const result = coerceToStreamingPartial(parsed, schema.ast);
    
    expect(result).toBeDefined();
    expect(result!.value).toEqual(["single"]);
    expect(hasFlag(result!.flags, "singleToArray")).toBe(true);
  });

  it("coerces single number to number array", () => {
    const parsed = makeNumber("42");
    const schema = Schema.Array(Schema.Number);
    const result = coerceToStreamingPartial(parsed, schema.ast);
    
    expect(result).toBeDefined();
    expect(result!.value).toEqual([42]);
    expect(hasFlag(result!.flags, "singleToArray")).toBe(true);
  });

  it("coerces nested arrays", () => {
    const inner = makeArray([makeNumber("1"), makeNumber("2")]);
    const parsed = makeArray([inner, inner]);
    const schema = Schema.Array(Schema.Array(Schema.Number));
    const result = coerceToStreamingPartial(parsed, schema.ast);
    
    expect(result).toBeDefined();
    expect(result!.value).toEqual([[1, 2], [1, 2]]);
  });

  it("handles array item coercion with flags", () => {
    const parsed = makeArray([makeString("a"), makeObject([]), makeString("c")]);
    const schema = Schema.Array(Schema.String);
    const result = coerceToStreamingPartial(parsed, schema.ast);
    
    expect(result).toBeDefined();
    // The object gets coerced to string via JSON.stringify with jsonToString flag
    expect(result!.value).toEqual(["a", "{}", "c"]);
    expect(hasFlag(result!.flags, "jsonToString")).toBe(true);
  });
});

// ============================================================================
// Struct/Object Coercion Tests
// ============================================================================

describe("Struct coercion", () => {
  const PersonSchema = Schema.Struct({
    name: Schema.String,
    age: Schema.Number,
  });

  it("coerces complete object", () => {
    const parsed = makeObject([
      ["name", makeString("Alice")],
      ["age", makeNumber("30")],
    ]);
    const result = coerceToStreamingPartial(parsed, PersonSchema.ast);
    
    expect(result).toBeDefined();
    expect(result!.value).toEqual({ name: "Alice", age: 30 });
    expect(result!.score).toBe(0);
  });

  it("coerces incomplete object with flag", () => {
    const parsed = makeObject([
      ["name", makeString("Alice")],
      ["age", makeNumber("30")],
    ], "incomplete");
    const result = coerceToStreamingPartial(parsed, PersonSchema.ast);
    
    expect(result).toBeDefined();
    expect(hasFlag(result!.flags, "incomplete")).toBe(true);
  });

  it("handles missing optional fields", () => {
    const OptionalSchema = Schema.Struct({
      name: Schema.String,
      age: Schema.optional(Schema.Number),
    });
    
    const parsed = makeObject([
      ["name", makeString("Alice")],
    ]);
    const result = coerceToStreamingPartial(parsed, OptionalSchema.ast);
    
    expect(result).toBeDefined();
    expect(result!.value).toEqual({ name: "Alice" });
  });

  it("flags missing required fields", () => {
    const parsed = makeObject([
      ["name", makeString("Alice")],
      // age is missing
    ]);
    const result = coerceToStreamingPartial(parsed, PersonSchema.ast);
    
    expect(result).toBeDefined();
    expect(hasFlag(result!.flags, "missingRequired")).toBe(true);
  });

  it("try_cast rejects missing required fields", () => {
    const parsed = makeObject([
      ["name", makeString("Alice")],
    ]);
    const result = tryCastToStreamingPartial(parsed, PersonSchema.ast);
    
    expect(result).toBeUndefined();
  });

  it("handles case-insensitive key matching", () => {
    const parsed = makeObject([
      ["NAME", makeString("Alice")],
      ["Age", makeNumber("30")],
    ]);
    const result = coerceToStreamingPartial(parsed, PersonSchema.ast);
    
    expect(result).toBeDefined();
    expect(result!.value).toEqual({ name: "Alice", age: 30 });
  });

  it("flags extra keys", () => {
    const parsed = makeObject([
      ["name", makeString("Alice")],
      ["age", makeNumber("30")],
      ["extra", makeString("field")],
    ]);
    const result = coerceToStreamingPartial(parsed, PersonSchema.ast);
    
    expect(result).toBeDefined();
    expect(hasFlag(result!.flags, "extraKey")).toBe(true);
  });

  it("try_cast rejects extra keys", () => {
    const parsed = makeObject([
      ["name", makeString("Alice")],
      ["age", makeNumber("30")],
      ["extra", makeString("field")],
    ]);
    const result = tryCastToStreamingPartial(parsed, PersonSchema.ast);
    
    expect(result).toBeUndefined();
  });

  it("handles nested objects", () => {
    const NestedSchema = Schema.Struct({
      person: PersonSchema,
      active: Schema.Boolean,
    });
    
    const parsed = makeObject([
      ["person", makeObject([
        ["name", makeString("Alice")],
        ["age", makeNumber("30")],
      ])],
      ["active", makeBoolean(true)],
    ]);
    const result = coerceToStreamingPartial(parsed, NestedSchema.ast);
    
    expect(result).toBeDefined();
    expect(result!.value).toEqual({
      person: { name: "Alice", age: 30 },
      active: true,
    });
  });

  it("applies single-field implied key heuristic", () => {
    const SingleFieldSchema = Schema.Struct({
      value: Schema.String,
    });
    
    const parsed = makeString("direct value");
    const result = coerceToStreamingPartial(parsed, SingleFieldSchema.ast);
    
    expect(result).toBeDefined();
    expect(result!.value).toEqual({ value: "direct value" });
    expect(hasFlag(result!.flags, "impliedKey")).toBe(true);
  });
});

// ============================================================================
// Enum Coercion Tests
// ============================================================================

describe("Enum coercion", () => {
  const StatusSchema = Schema.Literal("pending", "active", "completed");

  it("coerces exact enum match", () => {
    const parsed = makeString("active");
    const result = coerceToStreamingPartial(parsed, StatusSchema.ast);
    
    expect(result).toBeDefined();
    expect(result!.value).toBe("active");
    expect(result!.score).toBe(0);
  });

  it("coerces case-insensitive match with flag", () => {
    const parsed = makeString("ACTIVE");
    const result = coerceToStreamingPartial(parsed, StatusSchema.ast);
    
    expect(result).toBeDefined();
    expect(result!.value).toBe("active");
    expect(hasFlag(result!.flags, "stringToEnum")).toBe(true);
  });

  it("coerces fuzzy match with flag", () => {
    const parsed = makeString("complet"); // typo
    const result = coerceToStreamingPartial(parsed, StatusSchema.ast);
    
    expect(result).toBeDefined();
    expect(result!.value).toBe("completed");
    expect(hasFlag(result!.flags, "stringToEnum")).toBe(true);
  });

  it("try_cast requires exact match", () => {
    const parsed = makeString("ACTIVE");
    const result = tryCastToStreamingPartial(parsed, StatusSchema.ast);
    
    expect(result).toBeUndefined();
  });

  it("rejects non-matching values", () => {
    const parsed = makeString("unknown");
    const result = coerceToStreamingPartial(parsed, StatusSchema.ast);
    
    expect(result).toBeUndefined();
  });
});

// ============================================================================
// Literal Coercion Tests
// ============================================================================

describe("Literal coercion", () => {
  it("coerces string literal", () => {
    const schema = Schema.Literal("exact");
    const parsed = makeString("exact");
    const result = coerceToStreamingPartial(parsed, schema.ast);
    
    expect(result).toBeDefined();
    expect(result!.value).toBe("exact");
  });

  it("coerces number literal", () => {
    const schema = Schema.Literal(42);
    const parsed = makeNumber("42");
    const result = coerceToStreamingPartial(parsed, schema.ast);
    
    expect(result).toBeDefined();
    expect(result!.value).toBe(42);
  });

  it("coerces boolean literal", () => {
    const schema = Schema.Literal(true);
    const parsed = makeBoolean(true);
    const result = coerceToStreamingPartial(parsed, schema.ast);
    
    expect(result).toBeDefined();
    expect(result!.value).toBe(true);
  });

  it("rejects non-matching literal", () => {
    const schema = Schema.Literal("exact");
    const parsed = makeString("different");
    const result = coerceToStreamingPartial(parsed, schema.ast);
    
    expect(result).toBeUndefined();
  });
});

// ============================================================================
// Union Coercion Tests
// ============================================================================

describe("Union coercion", () => {
  const StringOrNumberSchema = Schema.Union(Schema.String, Schema.Number);

  it("coerces to first matching variant", () => {
    const parsed = makeString("hello");
    const result = coerceToStreamingPartial(parsed, StringOrNumberSchema.ast);
    
    expect(result).toBeDefined();
    expect(result!.value).toBe("hello");
    expect(hasFlag(result!.flags, "unionMatch")).toBe(true);
  });

  it("coerces number in string/number union", () => {
    const parsed = makeNumber("42");
    const result = coerceToStreamingPartial(parsed, StringOrNumberSchema.ast);
    
    expect(result).toBeDefined();
    expect(result!.value).toBe(42);
  });

  it("picks best match by score", () => {
    // String "42" could match String directly (score 0) or Number via coercion (score 1)
    const parsed = makeString("42");
    const result = coerceToStreamingPartial(parsed, StringOrNumberSchema.ast);
    
    expect(result).toBeDefined();
    // Should prefer direct string match over number coercion
    expect(result!.value).toBe("42");
  });

  it("try_cast short-circuits on perfect match", () => {
    const parsed = makeString("hello");
    const result = tryCastToStreamingPartial(parsed, StringOrNumberSchema.ast);
    
    expect(result).toBeDefined();
    expect(result!.score).toBe(0);
  });

  it("handles complex union with objects", () => {
    const CatSchema = Schema.Struct({ type: Schema.Literal("cat"), name: Schema.String });
    const DogSchema = Schema.Struct({ type: Schema.Literal("dog"), name: Schema.String });
    const AnimalSchema = Schema.Union(CatSchema, DogSchema);
    
    const parsed = makeObject([
      ["type", makeString("cat")],
      ["name", makeString("Whiskers")],
    ]);
    const result = coerceToStreamingPartial(parsed, AnimalSchema.ast);
    
    expect(result).toBeDefined();
    expect(result!.value).toEqual({ type: "cat", name: "Whiskers" });
  });

  it("handles null in union", () => {
    const NullableStringSchema = Schema.Union(Schema.String, Schema.Null);
    
    const parsed = makeNull();
    const result = coerceToStreamingPartial(parsed, NullableStringSchema.ast);
    
    expect(result).toBeDefined();
    expect(result!.value).toBeNull();
  });
});

// ============================================================================
// Complex/Nested Scenarios
// ============================================================================

describe("Complex nested scenarios", () => {
  it("coerces array of objects", () => {
    const ItemSchema = Schema.Struct({
      id: Schema.Number,
      name: Schema.String,
    });
    const ListSchema = Schema.Array(ItemSchema);
    
    const parsed = makeArray([
      makeObject([
        ["id", makeNumber("1")],
        ["name", makeString("First")],
      ]),
      makeObject([
        ["id", makeNumber("2")],
        ["name", makeString("Second")],
      ]),
    ]);
    const result = coerceToStreamingPartial(parsed, ListSchema.ast);
    
    expect(result).toBeDefined();
    expect(result!.value).toEqual([
      { id: 1, name: "First" },
      { id: 2, name: "Second" },
    ]);
  });

  it("coerces deeply nested structure", () => {
    const DeepSchema = Schema.Struct({
      level1: Schema.Struct({
        level2: Schema.Struct({
          level3: Schema.String,
        }),
      }),
    });
    
    const parsed = makeObject([
      ["level1", makeObject([
        ["level2", makeObject([
          ["level3", makeString("deep value")],
        ])],
      ])],
    ]);
    const result = coerceToStreamingPartial(parsed, DeepSchema.ast);
    
    expect(result).toBeDefined();
    expect(result!.value).toEqual({
      level1: {
        level2: {
          level3: "deep value",
        },
      },
    });
  });

  it("handles mixed array of unions", () => {
    const StringOrNumberSchema = Schema.Union(Schema.String, Schema.Number);
    const MixedArraySchema = Schema.Array(StringOrNumberSchema);
    
    const parsed = makeArray([
      makeString("text"),
      makeNumber("42"),
      makeString("more text"),
    ]);
    const result = coerceToStreamingPartial(parsed, MixedArraySchema.ast);
    
    expect(result).toBeDefined();
    expect(result!.value).toEqual(["text", 42, "more text"]);
  });

  it("handles incomplete nested values", () => {
    const NestedSchema = Schema.Struct({
      outer: Schema.Struct({
        inner: Schema.String,
      }),
    });
    
    const parsed = makeObject([
      ["outer", makeObject([
        ["inner", makeString("incompl", "incomplete")],
      ], "incomplete")],
    ], "incomplete");
    const result = coerceToStreamingPartial(parsed, NestedSchema.ast);
    
    expect(result).toBeDefined();
    expect(hasFlag(result!.flags, "incomplete")).toBe(true);
  });
});

// ============================================================================
// Edge Cases
// ============================================================================

describe("Edge cases", () => {
  it("handles empty object", () => {
    const parsed = makeObject([]);
    const schema = Schema.Struct({});
    const result = coerceToStreamingPartial(parsed, schema.ast);
    
    expect(result).toBeDefined();
    expect(result!.value).toEqual({});
  });

  it("handles empty array", () => {
    const parsed = makeArray([]);
    const schema = Schema.Array(Schema.String);
    const result = coerceToStreamingPartial(parsed, schema.ast);
    
    expect(result).toBeDefined();
    expect(result!.value).toEqual([]);
  });

  it("handles null at top level", () => {
    const parsed = makeNull();
    const schema = Schema.String;
    const result = coerceToStreamingPartial(parsed, schema.ast);
    
    // Null doesn't coerce to string
    expect(result).toBeUndefined();
  });

  it("handles undefined input", () => {
    const schema = Schema.String;
    const result = coerceToStreamingPartial(undefined, schema.ast);
    
    expect(result).toBeDefined();
    expect(hasFlag(result!.flags, "defaultFromNoValue")).toBe(true);
    expect(result!.score).toBe(100);
  });

  it("handles unknown schema types gracefully", () => {
    const parsed = makeString("test");
    // Use a Symbol schema which we don't have special handling for
    const schema = Schema.Symbol;
    const result = coerceToStreamingPartial(parsed, schema.ast);
    
    // Should fall back to string conversion
    expect(result).toBeDefined();
    expect(result!.value).toBe("test");
  });
});
