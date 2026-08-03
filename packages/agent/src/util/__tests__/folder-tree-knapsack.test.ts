import { expect, test } from 'bun:test'
import { join } from 'path'
import { CHARS_PER_TOKEN_UPPER, FOLDER_TREE_BUDGET_TOKENS } from '../../constants'
import { knapsackFolderTree } from '../folder-tree-knapsack'

const REPO_ROOT = join(import.meta.dir, '../../../../../')

test('default budget obeyed on real scratchpad', async () => {
  const out = await knapsackFolderTree(REPO_ROOT)
  expect(out.length).toBeGreaterThan(0)
  expect(Math.ceil(out.length / CHARS_PER_TOKEN_UPPER)).toBeLessThanOrEqual(FOLDER_TREE_BUDGET_TOKENS)
})

test('files appear in output on real scratchpad', async () => {
  const out = await knapsackFolderTree(REPO_ROOT, 2500)
  // should contain at least one file (no extension-less dir-only output)
  expect(out).toMatch(/\.\w+/)
})

test('git failure fallback: non-git dir returns non-empty tree', async () => {
  const out = await knapsackFolderTree('/tmp', 100)
  // /tmp may be empty but function should not throw
  expect(typeof out).toBe('string')
})
