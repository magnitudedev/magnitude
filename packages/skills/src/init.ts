import * as fs from 'fs'
import * as path from 'path'

/**
 * Initialize project skills by copying built-in skills to the project directory.
 * 
 * Built-in skills are located at:
 * - Compiled mode: Bun.embeddedFiles filtered by 'builtin/' prefix
 * - Dev mode: packages/skills/builtin/ directory (relative to this file)
 * 
 * Target: <cwd>/.magnitude/skills/
 * 
 * If the target directory exists and contains SKILL.md files, initialization
 * is skipped (user owns these skills).
 */
export async function initProjectSkills(cwd: string): Promise<void> {
  const targetDir = path.join(cwd, '.magnitude', 'skills')
  
  // Check if target exists and has SKILL.md files (user owns these)
  if (fs.existsSync(targetDir)) {
    const hasExistingSkills = fs.readdirSync(targetDir, { withFileTypes: true })
      .filter(d => d.isDirectory())
      .some(d => fs.existsSync(path.join(targetDir, d.name, 'SKILL.md')))
    
    if (hasExistingSkills) {
      console.log(`Skipping skill initialization: .magnitude/skills/ already exists with skills`)
      return
    }
  }
  
  // Get builtin skill files
  const builtinSkills = await readBuiltinSkills()
  
  if (builtinSkills.length === 0) {
    console.warn('No built-in skills found')
    return
  }
  
  // Create target directory
  fs.mkdirSync(targetDir, { recursive: true })
  
  // Copy each skill
  let copied = 0
  for (const skill of builtinSkills) {
    const skillDir = path.join(targetDir, skill.name)
    fs.mkdirSync(skillDir, { recursive: true })
    
    const targetPath = path.join(skillDir, 'SKILL.md')
    fs.writeFileSync(targetPath, skill.content, 'utf-8')
    copied++
  }
  
  console.log(`Initialized ${copied} skills in .magnitude/skills/`)
}

interface BuiltinSkill {
  name: string
  content: string
}

interface EmbeddedFile {
  name: string
  text(): Promise<string>
}

async function readBuiltinSkills(): Promise<BuiltinSkill[]> {
  // Check if we're in compiled mode (binary with embedded builtin files)
  // Note: Bun.embeddedFiles is truthy even in dev (empty array), so check length
  if (typeof Bun !== 'undefined' && Bun.embeddedFiles && Bun.embeddedFiles.length > 0) {
    const embeddedFiles = (Bun.embeddedFiles as unknown as EmbeddedFile[])
      .filter(f => f.name.includes('builtin/') && f.name.endsWith('SKILL.md'))
    
    const skills: BuiltinSkill[] = []
    for (const file of embeddedFiles) {
      // Path format: .../builtin/<name>/SKILL.md (prefix may vary)
      const match = file.name.match(/builtin\/([^/]+)\/SKILL\.md$/)
      if (!match) continue
      
      const name = match[1]
      const content = await file.text()
      skills.push({ name, content })
    }
    
    return skills
  }
  
  // Dev mode: scan filesystem
  return readBuiltinSkillsFromFilesystem()
}

function readBuiltinSkillsFromFilesystem(): BuiltinSkill[] {
  // Resolve builtin/ directory relative to this source file.
  // import.meta.dirname is Bun-specific and correctly resolves to the
  // directory containing this file, even when imported from other packages.
  const builtinDir = path.join(import.meta.dirname!, '..', 'builtin')

  if (!fs.existsSync(builtinDir)) {
    console.warn(`Builtin skills directory not found: ${builtinDir}`)
    return []
  }
  
  const skills: BuiltinSkill[] = []
  
  for (const entry of fs.readdirSync(builtinDir, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue
    
    const skillPath = path.join(builtinDir, entry.name, 'SKILL.md')
    if (!fs.existsSync(skillPath)) continue
    
    const content = fs.readFileSync(skillPath, 'utf-8')
    skills.push({ name: entry.name, content })
  }
  
  return skills
}
