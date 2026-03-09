import { TypeBuilder } from '@boundaryml/baml/type_builder';
import { FieldType } from "@boundaryml/baml/native";

export interface JsonSchema {
  type?: string;
  title?: string;
  description?: string; // This description at the schema's top level is NOT used for Class or Enum descriptions by the Python script
  default?: any;

  properties?: Record<string, JsonSchema>;
  required?: string[];
  additionalProperties?: boolean | JsonSchema;

  items?: JsonSchema;

  enum?: any[];

  $ref?: string;

  anyOf?: JsonSchema[];

  [key: string]: any;
}

function randomName(): string {
  const chars = 'abcdefghijklmnopqrstuvwxyz';
  let result = '';
  for (let i = 0; i < 12; i++) {
    result += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return result;
}

class SchemaAdder {
  private tb: TypeBuilder;
  private rootSchema: JsonSchema;
  private refCache: Record<string, FieldType> = {};
  private makeAllOptional: boolean;

  constructor(tb: TypeBuilder, rootSchema: JsonSchema, makeAllOptional: boolean = false) {
    this.tb = tb;
    this.rootSchema = rootSchema;
    this.makeAllOptional = makeAllOptional;
  }

  private _parseObject(jsonSchema: JsonSchema): FieldType {
    if (jsonSchema.type !== "object") {
      throw new Error(`_parseObject called with non-object type: ${jsonSchema.type}`);
    }
    // Just make it always random?? else we get sometimes stuff like "Invalid schema provided to extract: Class Reaction already exists"
    let name = randomName();
    // let name = jsonSchema.title;
    // if (name === undefined || typeof name !== 'string') {
    //   // Name doesn't actually affect much if field used as return value anyway, just needs to be unique
    //   name = randomName();
    //   //throw new Error("Title (string) is required in JSON schema for object type that becomes a BAML class.");
    // }

    const newCls = this.tb.addClass(name);

    const requiredFields = jsonSchema.required || [];
    if (!Array.isArray(requiredFields) || !requiredFields.every(rf => typeof rf === 'string')) {
      throw new Error(`'required' property in object '${name}' must be an array of strings.`);
    }

    const properties = jsonSchema.properties;
    if (properties && typeof properties === 'object') {
      for (const [fieldName, fieldSchemaUntyped] of Object.entries(properties)) {
        if (typeof fieldSchemaUntyped !== 'object' || fieldSchemaUntyped === null) {
          console.warn(`Property schema for '${fieldName}' in class '${name}' is not a valid object/schema. Skipping.`);
          continue;
        }
        const fieldSchema = fieldSchemaUntyped as JsonSchema;

        const defaultValue = fieldSchema.default;
        let fieldType: FieldType;

        if (fieldSchema.properties === undefined && fieldSchema.type === "object") {
          console.warn(
            `Field '${fieldName}' in class '${name}' uses generic dict type (object without properties) ` +
            `which defaults to map<string, string>. ` +
            "If a more specific type is needed, please provide a specific schema with properties.",
          );
          fieldType = this.tb.map(this.tb.string(), this.tb.string());
        } else {
          fieldType = this.parse(fieldSchema);
        }

        if (!requiredFields.includes(fieldName) || this.makeAllOptional) {
          if (defaultValue === undefined || this.makeAllOptional) {
            fieldType = fieldType.optional();
          }
        }
        const propertyBuilder = newCls.addProperty(fieldName, fieldType);

        let fieldDescription = fieldSchema.description;
        if (typeof fieldDescription === 'string') {
          let finalDescription = fieldDescription.trim();
          if (defaultValue !== undefined) {
            finalDescription = `${finalDescription}\nDefault: ${JSON.stringify(defaultValue)}`;
            finalDescription = finalDescription.trim();
          }
          if (finalDescription.length > 0) {
            propertyBuilder.description(finalDescription);
          }
        } else if (defaultValue !== undefined) {
          const defaultDesc = `Default: ${JSON.stringify(defaultValue)}`;
          propertyBuilder.description(defaultDesc.trim());
        }
      }
    }
    return newCls.type();
  }

  private _parseString(jsonSchema: JsonSchema): FieldType {
    if (jsonSchema.type !== "string") {
      throw new Error(`_parseString called with non-string type: ${jsonSchema.type}`);
    }

    const title = jsonSchema.title;
    const enumValues = jsonSchema.enum;

    if (enumValues) {
      if (!Array.isArray(enumValues)) {
        throw new Error(`'enum' property for string type '${title || "anonymous"}' must be an array.`);
      }

      const stringEnumValues: string[] = [];
      for (const val of enumValues) {
        stringEnumValues.push(String(val));
      }

      if (title === undefined || typeof title !== 'string') {
        if (stringEnumValues.length === 0) {
          console.warn(`Anonymous enum (string type with 'enum' but no 'title') has no values. Defaulting to plain string type.`);
          return this.tb.string();
        }
        return this.tb.union(stringEnumValues.map(value => this.tb.literalString(value)));
      }

      const newEnum = this.tb.addEnum(title);

      if (stringEnumValues.length === 0) {
        console.warn(`Enum '${title}' has no values. An empty enum was created.`);
      }
      for (const value of stringEnumValues) {
        newEnum.addValue(value);
      }
      return newEnum.type();
    }
    return this.tb.string();
  }

  private _parseArray(jsonSchema: JsonSchema): FieldType {
    if (jsonSchema.type !== "array") {
      throw new Error(`_parseArray called with non-array type: ${jsonSchema.type}`);
    }

    const itemsSchema = jsonSchema.items;
    if (itemsSchema === undefined) {
      throw new Error(`Array field '${jsonSchema.title || "untitled array"}' is missing 'items' definition.`);
    }

    if (Array.isArray(itemsSchema) || typeof itemsSchema !== 'object' || itemsSchema === null) {
      throw new Error(`'items' property for array '${jsonSchema.title || "untitled array"}' must be a single schema object. Tuple/array types for 'items' are not supported by this converter (matching Python script's direct parsing of items).`);
    }

    return this.parse(itemsSchema as JsonSchema).list();
  }


  private _loadRef(ref: string): FieldType {
    if (!ref.startsWith("#/")) {
      throw new Error(`Only local references are supported: ${ref}`);
    }

    if (this.refCache[ref]) {
      return this.refCache[ref];
    }

    const pathParts = ref.substring(2).split('/');

    if (pathParts.length !== 2 || !pathParts[0] || !pathParts[1]) {
      throw new Error(`Unsupported $ref format: '${ref}'. Expected format like '#/collectionKey/definitionKey' matching Python logic.`);
    }

    const collectionKey = pathParts[0];
    const definitionKey = pathParts[1];

    const collection = (this.rootSchema as any)[collectionKey];
    if (typeof collection !== 'object' || collection === null) {
      throw new Error(`Reference collection '${collectionKey}' for $ref '${ref}' not found or not an object in the root schema.`);
    }

    if (!collection.hasOwnProperty(definitionKey)) {
      throw new Error(`Reference item '${definitionKey}' for $ref '${ref}' not found in collection '${collectionKey}'.`);
    }

    const targetSchema = collection[definitionKey];
    if (typeof targetSchema !== 'object' || targetSchema === null) {
      throw new Error(`Resolved $ref '${ref}' points to a non-object schema.`);
    }

    const fieldType = this.parse(targetSchema as JsonSchema);
    this.refCache[ref] = fieldType;
    return fieldType;
  }

  parse(jsonSchema: JsonSchema): FieldType {
    if (jsonSchema.$ref) {
      if (typeof jsonSchema.$ref !== 'string') {
        throw new Error("'$ref' property must be a string.");
      }
      return this._loadRef(jsonSchema.$ref);
    }

    // Handle 'const' next, as it's a very specific assertion
    if (jsonSchema.const !== undefined) {
      const constValue = jsonSchema.const;
      if (typeof constValue === 'string') {
        return this.tb.literalString(constValue);
      } else if (typeof constValue === 'number') {
        if (Number.isInteger(constValue)) {
          return this.tb.literalInt(constValue);
        } else {
          // BAML does not appear to have a literalFloat, based on available information.
          // The 'const' value itself isn't captured as a literal in the BAML type for floats.
          console.warn(
            `JSON Schema 'const' with float value '${constValue}' encountered. ` +
            `BAML will use base type (float) as literalFloat is not available. Title: '${jsonSchema.title || "anonymous"}'`
          );
          return this.tb.float();
        }
      } else if (typeof constValue === 'boolean') {
        return this.tb.literalBool(constValue);
      } else if (constValue === null) {
        return this.tb.null();
      }
      if (typeof constValue === 'object' && (jsonSchema.type !== 'object' && jsonSchema.type !== 'array')) {
        console.warn(
          `JSON Schema 'const' with complex value '${JSON.stringify(constValue)}' ` +
          `encountered. Title: '${jsonSchema.title || "anonymous"}'. ` +
          `Expected 'type: "object"' or 'type: "array"' if this const is to be further processed as such. ` +
          `Falling back to standard type parsing based on 'jsonSchema.type' (if present).`
        );
      }
    }

    if (jsonSchema.anyOf) {
      if (!Array.isArray(jsonSchema.anyOf)) {
        throw new Error("'anyOf' property must be an array of schema objects.");
      }
      if (!jsonSchema.anyOf.every(s => typeof s === 'object' && s !== null)) {
        throw new Error("'anyOf' array must contain valid schema objects.");
      }
      if (jsonSchema.anyOf.length === 0) {
        console.warn("'anyOf' is an empty array. Defaulting to string (Python script has no specific handling).");
        return this.tb.string();
      }
      return this.tb.union(jsonSchema.anyOf.map(subSchema => this.parse(subSchema as JsonSchema)));
    }

    const additionalProperties = jsonSchema.additionalProperties;
    if (additionalProperties && typeof additionalProperties === 'object' && (additionalProperties as JsonSchema).anyOf) {
      const apSchema = additionalProperties as JsonSchema;
      const anyOfInAp = apSchema.anyOf;
      if (!Array.isArray(anyOfInAp)) {
        throw new Error("'anyOf' in 'additionalProperties' must be an array of schema objects.");
      }
      if (!anyOfInAp.every(s => typeof s === 'object' && s !== null)) {
        throw new Error("'anyOf' in 'additionalProperties' must contain valid schema objects.");
      }
      if (anyOfInAp.length === 0) {
        console.warn("'anyOf' in 'additionalProperties' is empty. Defaulting map value to string.");
        return this.tb.map(this.tb.string(), this.tb.string());
      }
      const valueType = this.tb.union(anyOfInAp.map(subSchema => this.parse(subSchema as JsonSchema)));
      return this.tb.map(this.tb.string(), valueType);
    }

    const type = jsonSchema.type;

    if (type === undefined) {
      console.warn("Type field is missing in JSON schema, defaulting to string (Python script behavior).");
      return this.tb.string();
    }

    if (typeof type !== 'string') {
      throw new Error(`Unsupported 'type' format: Expected a string, got ${typeof type} (${JSON.stringify(type)}). Python script would raise ValueError.`);
    }

    let fieldType: FieldType;
    switch (type) {
      case "string":
        fieldType = this._parseString(jsonSchema);
        break;
      case "number":
        fieldType = this.tb.float();
        break;
      case "integer":
        fieldType = this.tb.int();
        break;
      case "object":
        fieldType = this._parseObject(jsonSchema);
        break;
      case "array":
        fieldType = this._parseArray(jsonSchema);
        break;
      case "boolean":
        fieldType = this.tb.bool();
        break;
      case "null":
        fieldType = this.tb.null();
        break;
      default:
        throw new Error(`Unsupported JSON Schema type: '${type}'`);
    }
    
    // Handle nullable types
    if (jsonSchema.nullable === true) {
      fieldType = fieldType.optional();
    }
    
    // Apply makeAllOptional logic to all field types, not just object properties
    if (this.makeAllOptional) {
      fieldType = fieldType.optional();
    }
    
    return fieldType;
  }
}

export function convertJsonSchemaToBaml(tb: TypeBuilder, jsonSchema: JsonSchema, makeAllOptional: boolean = false): FieldType {
  if (typeof jsonSchema !== 'object' || jsonSchema === null) {
    throw new Error("Invalid JSON schema provided. Must be an object.");
  }
  const parser = new SchemaAdder(tb, jsonSchema, makeAllOptional);
  return parser.parse(jsonSchema);
}
