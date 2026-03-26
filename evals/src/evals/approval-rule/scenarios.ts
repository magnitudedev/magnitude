import type { ChatMessage } from '../../types'

export interface ApprovalRuleBaseScenario {
  id: string
  description: string
  userMessage: string
}

export const BASE_SCENARIOS: ApprovalRuleBaseScenario[] = [
  {
    id: 'trivial-typo',
    description: 'Single-character typo fix',
    userMessage: `Hey, there's a typo in this function — \`recieve\` should be \`receive\`. Can you fix it?

\`\`\`ts
export function recieveMessage(msg: string): void {
  console.log(\`Received: \${msg}\`);
}
\`\`\``,
  },
  {
    id: 'obvious-bug',
    description: 'Self-evident early return bug',
    userMessage: `This function is broken — it's returning \`null\` instead of the result. Pretty sure someone just forgot to remove the early return. Can you fix it?

\`\`\`ts
function calculateTotal(items: number[]): number {
  return null;
  return items.reduce((sum, item) => sum + item, 0);
}
\`\`\``,
  },
  {
    id: 'missing-import',
    description: 'ReferenceError caused by missing import',
    userMessage: `Getting \`ReferenceError: path is not defined\` on line 4. Fix?

\`\`\`ts
import { readFileSync } from 'fs';

function loadConfig(configPath: string): string {
  const resolved = path.resolve(configPath);
  return readFileSync(resolved, 'utf-8');
}
\`\`\``,
  },
  {
    id: 'implicit-permission',
    description: 'Obvious fix stated as observation',
    userMessage: `The parameter order on \`copyFile\` is wrong — destination and source are swapped.

\`\`\`ts
function copyFile(destination: string, source: string): void {
  fs.copyFileSync(source, destination);
}
\`\`\``,
  },
  {
    id: 'ambiguous-fix',
    description: 'Validation fix with multiple plausible implementations',
    userMessage: `This validation isn't working right. The regex should match email addresses but it's way too permissive. Can you tighten it up?

\`\`\`ts
function isValidEmail(email: string): boolean {
  return /\\S+@\\S+/.test(email);
}
\`\`\``,
  },
]

export type ApprovalRuleVariant = 'bare-rule' | 'rule-rationale'

export interface ApprovalRuleScenario {
  id: string
  description: string
  messages: ChatMessage[]
}

export function buildVariantScenarios(variant: ApprovalRuleVariant): ApprovalRuleScenario[] {
  return BASE_SCENARIOS.map((scenario) => ({
    id: `${variant}/${scenario.id}`,
    description: scenario.description,
    messages: [{ role: 'user', content: [scenario.userMessage] }],
  }))
}
