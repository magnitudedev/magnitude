import { describe, expect, test } from 'bun:test'
import { observeOutput, renderOutputParts } from '../output-query'
import { buildOutputTree, type OutputNode } from '../output-tree'

describe('multimodal output', () => {
  const imageNode: OutputNode = {
    tag: 'image',
    base64: 'dGVzdA==',
    mediaType: 'image/png',
    width: 100,
    height: 100,
  }

  const imageTree: OutputNode = {
    tag: 'element',
    name: 'fs-view',
    attrs: {},
    children: [imageNode],
  }

  const mixedTree: OutputNode = {
    tag: 'element',
    name: 'tool',
    attrs: {},
    children: [
      { tag: 'element', name: 'url', attrs: {}, children: [{ tag: 'text', value: 'http://localhost' }] },
      { tag: 'element', name: 'screenshot', attrs: {}, children: [imageNode] },
      { tag: 'element', name: 'time', attrs: {}, children: [{ tag: 'text', value: '342' }] },
    ],
  }

  test('observeOutput with . returns children only', () => {
    expect(observeOutput(imageTree, '.')).toEqual([
      { type: 'image', base64: 'dGVzdA==', mediaType: 'image/png', width: 100, height: 100 },
    ])
  })

  test('observeOutput with non-. query on image tree returns selected image-containing element', () => {
    expect(observeOutput(imageTree, '//image')).toEqual([
      { type: 'image', base64: 'dGVzdA==', mediaType: 'image/png', width: 100, height: 100 },
    ])
  })

  test('observe with XPath selecting image-containing element', () => {
    const parts = observeOutput(mixedTree, '//screenshot')
    expect(parts.some(p => p.type === 'image')).toBe(true)
  })

  test('observe with XPath selecting text-only sibling of image', () => {
    expect(observeOutput(mixedTree, '//url')).toEqual([
      { type: 'text', text: 'http://localhost' },
    ])
  })

  test('observe with XPath selecting image element directly', () => {
    const parts = observeOutput(mixedTree, '//image')
    expect(parts.some(p => p.type === 'image')).toBe(true)
  })

  test('renderOutputParts on image node renders image content part', () => {
    expect(renderOutputParts(imageNode)).toEqual([
      { type: 'image', base64: 'dGVzdA==', mediaType: 'image/png', width: 100, height: 100 },
    ])
  })

  test('renderOutputParts on mixed text and image element preserves order', () => {
    const mixed: OutputNode = {
      tag: 'element',
      name: 'view',
      attrs: {},
      children: [
        { tag: 'text', value: 'before' },
        imageNode,
        { tag: 'text', value: 'after' },
      ],
    }

    expect(renderOutputParts(mixed)).toEqual([
      { type: 'text', text: '<view>before' },
      { type: 'image', base64: 'dGVzdA==', mediaType: 'image/png', width: 100, height: 100 },
      { type: 'text', text: 'after</view>' },
    ])
  })


  test('buildOutputTree with image value creates root element with image child', () => {
    expect(buildOutputTree('view', {
      base64: 'dGVzdA==',
      mediaType: 'image/png',
      width: 100,
      height: 100,
    }, { type: 'tag' })).toEqual({
      tag: 'element',
      name: 'view',
      attrs: {},
      children: [imageNode],
    })
  })

  test('buildOutputTree with nested image body in struct preserves nested image', () => {
    const tree = buildOutputTree('result', {
      screenshot: {
        base64: 'dGVzdA==',
        mediaType: 'image/png',
        width: 100,
        height: 100,
      },
    }, {
      type: 'tag',
      childTags: [{ tag: 'screenshot', field: 'screenshot' }],
    })

    expect(tree).toEqual({
      tag: 'element',
      name: 'result',
      attrs: {},
      children: [
        {
          tag: 'element',
          name: 'screenshot',
          attrs: {},
          children: [imageNode],
        },
      ],
    })
    expect(observeOutput(tree, '.').some(part => part.type === 'image')).toBe(true)
  })
})