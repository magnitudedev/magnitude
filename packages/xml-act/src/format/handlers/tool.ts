import { emit, pop, push, replace } from '../ops'
import { appendTopProse, endTopProse } from '../prose'
import { rawOpenTag, rawSelfCloseTag } from '../raw'
import type { Fx } from '../ops'
import type { TagSchema } from '../../execution/binding-validator'
import { validateChildAttr, validateToolAttr } from '../validate-attrs'
import type { AttributeValue, ParsedChild, ParsedElement, TagMap } from '../types'
import type { TagHandler, XmlActEvent, XmlActFrame } from '../types'
import { findFrame } from '../types'

function validateAttrs(
  tag: string,
  id: string,
  attrs: ReadonlyMap<string, string>,
  schema: TagSchema | undefined,
): { readonly valid: ReadonlyMap<string, AttributeValue>; readonly errors: readonly XmlActEvent[] } {
  const out = new Map<string, AttributeValue>()
  const errors: XmlActEvent[] = []
  for (const [key, value] of attrs) {
    if (!schema) {
      out.set(key, value)
      continue
    }
    const result = validateToolAttr(tag, schema, key, value)
    if (result.ok) {
      out.set(key, result.value)
    } else {
      if (
        result.error._tag === 'UnknownAttribute'
        && schema.children.has(key)
        && !schema.attributes.has(key)
      ) {
        // Allow attr→childTag normalization path: accept and do not emit UnknownAttribute.
        out.set(key, value)
      } else if (result.error._tag === 'UnknownAttribute') {
        errors.push({
          _tag: 'ParseError',
          error: { _tag: 'UnknownAttribute', id, tagName: tag, attribute: key, detail: `Unknown attribute '${key}' on <${tag}>` },
        })
      } else if (result.error._tag === 'InvalidAttributeValue') {
        errors.push({
          _tag: 'ParseError',
          error: {
            _tag: 'InvalidAttributeValue',
            id,
            tagName: tag,
            attribute: key,
            expected: result.error.expected,
            received: result.error.received,
            detail: `Invalid value for attribute '${key}' on <${tag}>: "${value}"`,
          },
        })
      }
    }
  }
  return { valid: out, errors }
}

export function toolHandler(
  tag: string,
  childTags: ReadonlySet<string>,
  schema: TagSchema | undefined,
  tags: TagMap,
): TagHandler<XmlActFrame, XmlActEvent> {
  return {
    open(ctx) {
      const top = ctx.stack[ctx.stack.length - 1]
      if (!ctx.afterNewline && top?.type === 'prose') {
        return appendTopProse(ctx.stack, rawOpenTag(tag, ctx.attrs))
      }
      const id = ctx.generateId()
      const { valid, errors } = validateAttrs(tag, id, ctx.attrs, schema)
      return [
        ...endTopProse(ctx.stack),
        ...errors.map((e) => ({ type: 'emit', event: e } as const)),
        emit({ _tag: 'TagOpened', tagName: tag, toolCallId: id, attributes: valid }),
        push({
          type: 'tool-body',
          tag,
          id,
          attrs: valid,
          body: '',
          children: [],
          childCounts: new Map(),
          childTags,
          schema,
          tags,
        }),
      ]
    },
    close(ctx) {
      const top = ctx.stack[ctx.stack.length - 1]
      if (top?.type === 'child-body' && top.parentTag === tag) {
        const parent = findFrame(ctx.stack, 'tool-body')
        if (!parent) return [pop]
        const parsed: ParsedChild = {
          tagName: top.childTagName,
          attributes: top.childAttrs,
          body: top.body,
        }
        const nextParent = { ...parent, children: [...parent.children, parsed] }
        const element: ParsedElement = {
          tagName: nextParent.tag,
          toolCallId: nextParent.id,
          attributes: nextParent.attrs,
          body: nextParent.body,
          children: nextParent.children,
        }
        return [
          emit({
            _tag: 'ChildComplete',
            parentToolCallId: top.parentToolId,
            childTagName: top.childTagName,
            childIndex: top.childIndex,
            attributes: top.childAttrs,
            body: top.body,
          }),
          pop,
          replace(nextParent),
          emit({ _tag: 'TagClosed', toolCallId: nextParent.id, tagName: nextParent.tag, element }),
          pop,
        ]
      }

      const frame = findFrame(ctx.stack, 'tool-body')
      if (!frame) return []
      const element: ParsedElement = {
        tagName: frame.tag,
        toolCallId: frame.id,
        attributes: frame.attrs,
        body: frame.body,
        children: frame.children,
      }
      return [emit({ _tag: 'TagClosed', toolCallId: frame.id, tagName: frame.tag, element }), pop]
    },
    selfClose(ctx) {
      const top = ctx.stack[ctx.stack.length - 1]
      if (!ctx.afterNewline && top?.type === 'prose') {
        return appendTopProse(ctx.stack, rawSelfCloseTag(tag, ctx.attrs))
      }
      const id = ctx.generateId()
      const { valid, errors } = validateAttrs(tag, id, ctx.attrs, schema)
      const element: ParsedElement = {
        tagName: tag,
        toolCallId: id,
        attributes: valid,
        body: '',
        children: [],
      }
      return [
        ...endTopProse(ctx.stack),
        ...errors.map((e) => ({ type: 'emit', event: e } as const)),
        emit({ _tag: 'TagOpened', tagName: tag, toolCallId: id, attributes: valid }),
        emit({ _tag: 'TagClosed', toolCallId: id, tagName: tag, element }),
      ]
    },
  }
}

export function childHandler(): TagHandler<XmlActFrame, XmlActEvent> {
  return {
    open(ctx) {
      const top = ctx.stack[ctx.stack.length - 1]
      if (top?.type === 'child-body') {
        const raw = rawOpenTag(ctx.tagName, ctx.attrs)
        return [
          replace({ ...top, body: top.body + raw }),
          emit({
            _tag: 'ChildBodyChunk',
            parentToolCallId: top.parentToolId,
            childTagName: top.childTagName,
            childIndex: top.childIndex,
            text: raw,
          }),
        ]
      }

      const parent = findFrame(ctx.stack, 'tool-body')
      if (!parent) return []
      const childIndex = parent.childCounts.get(ctx.tagName) ?? 0
      const childAttrs = new Map<string, AttributeValue>()
      const errors: XmlActEvent[] = []
      const childSchema = parent.schema?.children.get(ctx.tagName)
      for (const [k, v] of ctx.attrs) {
        if (!childSchema) {
          childAttrs.set(k, v)
          continue
        }
        const result = validateChildAttr(parent.tag, ctx.tagName, childSchema, k, v)
        if (result.ok) {
          childAttrs.set(k, result.value)
        } else if (result.error._tag === 'InvalidAttributeValue') {
          errors.push({
            _tag: 'ParseError',
            error: {
              _tag: 'InvalidAttributeValue',
              id: parent.id,
              tagName: parent.tag,
              attribute: k,
              expected: result.error.expected,
              received: result.error.received,
              detail: `Attribute '${k}' on child <${ctx.tagName}> inside <${parent.tag}> expects a ${result.error.expected}, got "${v}"`,
            },
          })
        }
      }
      const nextCounts = new Map(parent.childCounts)
      nextCounts.set(ctx.tagName, childIndex + 1)
      const childTags: TagMap = new Map([[ctx.tagName, childHandler()]])
      return [
        replace({ ...parent, childCounts: nextCounts }),
        ...errors.map((e) => ({ type: 'emit', event: e } as const)),
        emit({
          _tag: 'ChildOpened',
          parentToolCallId: parent.id,
          childTagName: ctx.tagName,
          childIndex,
          attributes: childAttrs,
        }),
        push({
          type: 'child-body',
          childTagName: ctx.tagName,
          childAttrs,
          body: '',
          parentToolId: parent.id,
          parentTag: parent.tag,
          childIndex,
          tags: childTags,
        }),
      ]
    },
    close(ctx) {
      const child = findFrame(ctx.stack, 'child-body')
      if (!child) {
        // No active child-body — this is a parent tool close (same tag name collision).
        // Delegate to tool close logic: emit TagClosed with full element, then pop.
        const frame = findFrame(ctx.stack, 'tool-body')
        if (!frame) return [pop]
        const element: ParsedElement = {
          tagName: frame.tag,
          toolCallId: frame.id,
          attributes: frame.attrs,
          body: frame.body,
          children: frame.children,
        }
        return [emit({ _tag: 'TagClosed', toolCallId: frame.id, tagName: frame.tag, element }), pop]
      }
      const parent = findFrame(ctx.stack, 'tool-body')
      if (!parent) return [pop]
      return [...completeChild(parent, child)]
    },
    selfClose(ctx) {
      const parent = findFrame(ctx.stack, 'tool-body')
      if (!parent) return []
      const childIndex = parent.childCounts.get(ctx.tagName) ?? 0
      const attrs = new Map<string, AttributeValue>()
      const errors: XmlActEvent[] = []
      const childSchema = parent.schema?.children.get(ctx.tagName)
      for (const [k, v] of ctx.attrs) {
        if (!childSchema) {
          attrs.set(k, v)
          continue
        }
        const result = validateChildAttr(parent.tag, ctx.tagName, childSchema, k, v)
        if (result.ok) {
          attrs.set(k, result.value)
        } else if (result.error._tag === 'InvalidAttributeValue') {
          errors.push({
            _tag: 'ParseError',
            error: {
              _tag: 'InvalidAttributeValue',
              id: parent.id,
              tagName: parent.tag,
              attribute: k,
              expected: result.error.expected,
              received: result.error.received,
              detail: `Attribute '${k}' on child <${ctx.tagName}> inside <${parent.tag}> expects a ${result.error.expected}, got "${v}"`,
            },
          })
        }
      }
      return [
        ...errors.map((e) => ({ type: 'emit', event: e } as const)),
        emit({
          _tag: 'ChildOpened',
          parentToolCallId: parent.id,
          childTagName: ctx.tagName,
          childIndex,
          attributes: attrs,
        }),
        emit({
          _tag: 'ChildComplete',
          parentToolCallId: parent.id,
          childTagName: ctx.tagName,
          childIndex,
          attributes: attrs,
          body: '',
        }),
      ]
    },
  }
}

export function completeChild(parent: Extract<XmlActFrame, { type: 'tool-body' }>, child: Extract<XmlActFrame, { type: 'child-body' }>): readonly Fx[] {
  const parsed: ParsedChild = {
    tagName: child.childTagName,
    attributes: child.childAttrs,
    body: child.body,
  }
  return [
    emit({
      _tag: 'ChildComplete',
      parentToolCallId: child.parentToolId,
      childTagName: child.childTagName,
      childIndex: child.childIndex,
      attributes: child.childAttrs,
      body: child.body,
    }),
    pop,
    replace({ ...parent, children: [...parent.children, parsed] }),
  ]
}