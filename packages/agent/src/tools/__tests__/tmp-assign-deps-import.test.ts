import { test } from 'bun:test'

test('import prompts', async () => {
  await import('../../prompts')
})

test('import tasks index', async () => {
  await import('../../tasks/index')
})

test('import agents', async () => {
  await import('../../agents')
})

test('import assign module', async () => {
  await import('../../tasks/operations/assign')
})
