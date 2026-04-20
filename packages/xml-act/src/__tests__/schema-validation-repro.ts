import { Schema } from '@effect/schema'

// Exact same schema as create-task tool
const CreateTaskInputSchema = Schema.Struct({
  id: Schema.String,
  title: Schema.String,
  parent: Schema.optional(Schema.String),
})

// Test 1: With empty parent string (what the Mact parser produces)
const input1 = { id: 'dummy-task', title: 'Dummy task for testing', parent: '' }
console.log('Test 1: parent=""')
try {
  const result = Schema.decodeUnknownSync(CreateTaskInputSchema)(input1)
  console.log('  PASS:', JSON.stringify(result))
} catch (e) {
  console.log('  FAIL:', e instanceof Error ? e.message : String(e))
}

// Test 2: Without parent field (what it should be for no parent)
const input2 = { id: 'dummy-task', title: 'Dummy task for testing' }
console.log('\nTest 2: no parent field')
try {
  const result = Schema.decodeUnknownSync(CreateTaskInputSchema)(input2)
  console.log('  PASS:', JSON.stringify(result))
} catch (e) {
  console.log('  FAIL:', e instanceof Error ? e.message : String(e))
}

// Test 3: With parent = undefined
const input3 = { id: 'dummy-task', title: 'Dummy task for testing', parent: undefined }
console.log('\nTest 3: parent=undefined')
try {
  const result = Schema.decodeUnknownSync(CreateTaskInputSchema)(input3)
  console.log('  PASS:', JSON.stringify(result))
} catch (e) {
  console.log('  FAIL:', e instanceof Error ? e.message : String(e))
}

// Now test what the tool execute function does with each input
console.log('\n=== Tool execute behavior ===')

for (const [label, input] of [
  ['parent=""', input1],
  ['no parent', input2],
  ['parent=undefined', input3],
] as const) {
  const parentId = (input as any).parent ?? null
  console.log(`\n${label}:`)
  console.log(`  input.parent = ${JSON.stringify((input as any).parent)}`)
  console.log(`  parentId = input.parent ?? null = ${JSON.stringify(parentId)}`)
  console.log(`  Boolean(parentId) = ${Boolean(parentId)}`)
  console.log(`  if (directive.parentId) → ${parentId ? 'TRUE (looks up parent)' : 'FALSE (no parent)'}`)
}
