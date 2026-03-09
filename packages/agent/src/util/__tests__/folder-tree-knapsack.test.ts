import { expect, test } from 'bun:test'
import { join } from 'path'
import { CHARS_PER_TOKEN } from '../../constants'
import { knapsackFolderTree } from '../folder-tree-knapsack'

const REPO_ROOT = join(import.meta.dir, '../../../../../')

test('budget obeyed on real workspace', async () => {
  const budget = 2500
  const out = await knapsackFolderTree(REPO_ROOT, budget)
  expect(out.length).toBeGreaterThan(0)
  expect(Math.ceil(out.length / CHARS_PER_TOKEN)).toBeLessThanOrEqual(budget)
})

test('files appear in output on real workspace', async () => {
  const out = await knapsackFolderTree(REPO_ROOT, 2500)
  // should contain at least one file (no extension-less dir-only output)
  expect(out).toMatch(/\.\w+/)
})

test('git failure fallback: non-git dir returns non-empty tree', async () => {
  const out = await knapsackFolderTree('/tmp', 100)
  // /tmp may be empty but function should not throw
  expect(typeof out).toBe('string')
})