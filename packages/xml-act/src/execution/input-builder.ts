/**
 * InputBuilder — convert ParsedElement + XmlTagBinding into a tool input object.
 *
 * Maps XML structure to the tool's expected input shape based on the binding.
 */

import type { ParsedElement } from '../parser/types'
import type { XmlTagBinding } from '../types'

/** Set a value at a dotted path, creating intermediate objects as needed. */
function setNestedValue(obj: Record<string, unknown>, path: string, value: unknown): void {
  const segments = path.split('.')
  let current = obj
  for (let i = 0; i < segments.length - 1; i++) {
    if (!(segments[i] in current)) {
      current[segments[i]] = {}
    }
    current = current[segments[i]] as Record<string, unknown>
  }
  current[segments[segments.length - 1]] = value
}

/**
 * Build a tool input object from a parsed XML element and its binding.
 */
export function buildInput(element: ParsedElement, binding: XmlTagBinding): Record<string, unknown> {
  const input: Record<string, unknown> = {}

  // Attributes → input fields
  if (binding.attributes) {
    for (const attrName of binding.attributes) {
      const value = element.attributes.get(attrName)
      if (value !== undefined) {
        input[attrName] = value
      }
    }
  }

  // Body → input field
  if (binding.body) {
    input[binding.body] = element.body.trim()
  }

  // ChildTags → scalar fields (fixed named child elements)
  if (binding.childTags) {
    for (const ct of binding.childTags) {
      const xmlTag = ct.tag
      const child = element.children.find(c => c.tagName === xmlTag)
      if (child) {
        setNestedValue(input, ct.field, child.body.replace(/^\n/, '').replace(/\n$/, ''))
      }
    }
  }

  // Children → array fields (repeated child elements)
  if (binding.children) {
    for (const childBinding of binding.children) {
      const childTag = childBinding.tag ?? childBinding.field
      const matchingChildren = element.children.filter(c => c.tagName === childTag)
      const entries: Record<string, unknown>[] = []

      for (const child of matchingChildren) {
        const entry: Record<string, unknown> = {}

        // Child attributes
        if (childBinding.attributes) {
          for (const attrName of childBinding.attributes) {
            const value = child.attributes.get(attrName)
            if (value !== undefined) {
              entry[attrName] = value
            }
          }
        }

        // Child body
        if (childBinding.body) {
          entry[childBinding.body] = child.body.trim()
        }

        entries.push(entry)
      }

      input[childBinding.field] = entries
    }
  }

  // ChildRecord → record field (repeated elements with key attr)
  if (binding.childRecord) {
    const { field, tag: childTag, keyAttr } = binding.childRecord
    const matchingChildren = element.children.filter(c => c.tagName === childTag)
    const record: Record<string, string> = {}

    for (const child of matchingChildren) {
      const key = child.attributes.get(keyAttr)
      if (key !== undefined) {
        record[String(key)] = child.body.trim()
      }
    }

    input[field] = record
  }

  return input
}

