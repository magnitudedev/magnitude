import { emit, pop, push, replace } from '../ops'
import { appendTopProse, endTopProse } from '../prose'
import { rawCloseTag, rawOpenTag } from '../raw'
import type { TagMap, TagHandler, XmlActFrame, XmlActEvent } from '../types'
import { findFrame } from '../types'

export function thinkHandler(plainThinkTags: TagMap): TagHandler<XmlActFrame, XmlActEvent> {
  return {
    open(ctx) {
      const raw = rawOpenTag(ctx.tagName, ctx.attrs)
      if (!ctx.afterNewline) {
        const currentThink = findFrame(ctx.stack, 'think')
        if (currentThink && currentThink.tag === ctx.tagName) {
          return [
            replace({ ...currentThink, body: currentThink.body + raw }),
            emit({ _tag: 'ProseChunk', patternId: 'think', text: raw }),
          ]
        }
        return appendTopProse(ctx.stack, raw)
      }

      const currentThink = findFrame(ctx.stack, 'think')
      if (currentThink && currentThink.tag === ctx.tagName) {
        return [
          replace({ ...currentThink, depth: currentThink.depth + 1, body: currentThink.body + raw }),
          emit({ _tag: 'ProseChunk', patternId: 'think', text: raw }),
        ]
      }

      return [
        ...endTopProse(ctx.stack),
        push({
          type: 'think',
          tag: ctx.tagName,
          body: '',
          depth: 0,
          about: ctx.attrs.get('about') ?? null,
          activeLens: null,
          lenses: [],
          tags: plainThinkTags,
        }),
      ]
    },
    close(ctx) {
      const think = findFrame(ctx.stack, 'think')
      if (!think) return []
      const rawClose = rawCloseTag(ctx.tagName)

      if (!ctx.afterNewline) {
        return [replace({ ...think, body: think.body + rawClose }), emit({ _tag: 'ProseChunk', patternId: 'think', text: rawClose })]
      }

      if (think.depth > 0) {
        return [replace({ ...think, depth: think.depth - 1, body: think.body + rawClose }), emit({ _tag: 'ProseChunk', patternId: 'think', text: rawClose })]
      }

      return [emit({ _tag: 'ProseEnd', patternId: 'think', content: think.body, about: think.about }), pop]
    },
    selfClose() {
      return []
    },
  }
}

export function lensHandler(
  betweenLensTags: TagMap,
  insideLensTags: TagMap,
): TagHandler<XmlActFrame, XmlActEvent> {
  return {
    open(ctx) {
      const think = findFrame(ctx.stack, 'think')
      const nextName = ctx.attrs.get('name') ?? ''

      if (think?.tag === 'lens' && think.activeLens) {
        const raw = rawOpenTag('lens', ctx.attrs)
        return [
          replace({
            ...think,
            activeLens: {
              ...think.activeLens,
              body: think.activeLens.body + raw,
              depth: think.activeLens.depth + 1,
            },
          }),
          emit({ _tag: 'LensChunk', text: raw }),
        ]
      }

      return [
        ...endTopProse(ctx.stack),
        emit({ _tag: 'LensStart', name: nextName }),
        push({
          type: 'think',
          tag: 'lens',
          body: '',
          depth: 0,
          about: null,
          activeLens: { name: nextName, body: '', depth: 0 },
          lenses: [],
          tags: insideLensTags,
        }),
      ]
    },
    close(ctx) {
      const think = findFrame(ctx.stack, 'think')
      if (think?.tag !== 'lens' || !think.activeLens) return []
      if (think.activeLens.depth > 0) {
        const raw = rawCloseTag('lens')
        return [
          replace({
            ...think,
            activeLens: {
              ...think.activeLens,
              body: think.activeLens.body + raw,
              depth: think.activeLens.depth - 1,
            },
          }),
          emit({ _tag: 'LensChunk', text: raw }),
        ]
      }
      const content = think.activeLens.body.trim()
      return [emit({ _tag: 'LensEnd', name: think.activeLens.name, content }), pop]
    },
    selfClose(ctx) {
      const name = ctx.attrs.get('name') ?? ''
      return [...endTopProse(ctx.stack), emit({ _tag: 'LensStart', name }), emit({ _tag: 'LensEnd', name, content: '' })]
    },
  }
}
