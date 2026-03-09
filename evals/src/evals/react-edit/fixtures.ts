/**
 * Load react-edit-benchmark fixtures from the extracted fixtures/ directory.
 */
import { readdirSync, readFileSync, existsSync } from 'fs'
import { join, basename } from 'path'

export interface FixtureMetadata {
  mutationType: string
  mutationCategory: string
  difficulty: 'easy' | 'medium' | 'hard' | 'nightmare'
  difficultyScore: number
  lineNumber: number
  originalSnippet: string
  mutatedSnippet: string
  filePath: string
}

export interface EditFixture {
  id: string
  prompt: string
  metadata: FixtureMetadata
  /** filename → content */
  inputFiles: Record<string, string>
  /** filename → content */
  expectedFiles: Record<string, string>
  fixturePath: string
}

const FIXTURES_DIR = join(import.meta.dir, '../../../fixtures/react-edit')

function listFiles(dir: string): string[] {
  if (!existsSync(dir)) return []
  return readdirSync(dir, { withFileTypes: true })
    .filter(e => e.isFile())
    .map(e => e.name)
    .sort()
}

function parseMetadata(raw: Record<string, unknown>): FixtureMetadata {
  return {
    mutationType: (raw.mutation_type ?? raw.mutationType) as string,
    mutationCategory: (raw.mutation_category ?? raw.mutationCategory ?? raw.category) as string,
    difficulty: (raw.difficulty as FixtureMetadata['difficulty']),
    difficultyScore: (raw.difficulty_score ?? raw.difficultyScore) as number,
    lineNumber: (raw.line_number ?? raw.lineNumber) as number,
    originalSnippet: (raw.original_snippet ?? raw.originalSnippet) as string,
    mutatedSnippet: (raw.mutated_snippet ?? raw.mutatedSnippet) as string,
    filePath: (raw.file_path ?? raw.filePath) as string,
  }
}

export function loadFixtures(): EditFixture[] {
  if (!existsSync(FIXTURES_DIR)) {
    throw new Error(
      `Fixtures not found at ${FIXTURES_DIR}. ` +
      `Run: cd evals && tar xzf ../reference/oh-my-pi/packages/react-edit-benchmark/fixtures.tar.gz -C .`
    )
  }

  const entries = readdirSync(FIXTURES_DIR, { withFileTypes: true })
    .filter(e => e.isDirectory())
    .map(e => e.name)
    .sort()

  return entries.map(id => {
    const fixturePath = join(FIXTURES_DIR, id)
    const prompt = readFileSync(join(fixturePath, 'prompt.md'), 'utf-8').trim()
    const raw = JSON.parse(readFileSync(join(fixturePath, 'metadata.json'), 'utf-8'))
    const metadata = parseMetadata(raw)

    const inputDir = join(fixturePath, 'input')
    const expectedDir = join(fixturePath, 'expected')

    const inputFiles: Record<string, string> = {}
    for (const f of listFiles(inputDir)) {
      inputFiles[f] = readFileSync(join(inputDir, f), 'utf-8')
    }

    const expectedFiles: Record<string, string> = {}
    for (const f of listFiles(expectedDir)) {
      expectedFiles[f] = readFileSync(join(expectedDir, f), 'utf-8')
    }

    return { id, prompt, metadata, inputFiles, expectedFiles, fixturePath }
  })
}
