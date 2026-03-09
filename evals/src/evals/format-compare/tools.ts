/**
 * Fake tool definitions for format comparison eval.
 * These define the tools that all format variants describe to the model.
 */

export interface FakeTool {
  name: string
  description: string
  params: { name: string; type: string; description: string; required?: boolean }[]
  xmlBinding?: {
    tagName: string
    attributes?: string[]
    body?: string
    selfClosing?: boolean
    children?: { tag: string; attributes?: string[]; body?: string }[]
  }
  /** Example return value (XML string) for simulated responses */
  returnType?: string
  exampleReturn?: string
}

export const FAKE_TOOLS: FakeTool[] = [
  {
    name: 'fs.read',
    description: 'Read file content as string',
    params: [
      {
        name: 'path',
        type: 'string',
        required: true,
        description: 'Relative path from cwd',
      },
    ],
    returnType: 'string',
    xmlBinding: {
      tagName: 'fs-read',
      attributes: ['path'],
      selfClosing: true,
    },
    exampleReturn: `<result tool="fs.read">
  <file path="src/auth/login.ts">
import { db } from '../db';
import { compare } from 'bcryptjs';

export async function login(email: string, password: string) {
  const user = await db.user.findUnique({ where: { email } });
  if (!user) return { ok: false, error: 'Invalid credentials' };

  const valid = await compare(password, user.passwordHash);
  if (!valid) return { ok: false, error: 'Invalid credentials' };

  return { ok: true, userId: user.id };
}
</file>
</result>`,
  },
  {
    name: 'fs.write',
    description: 'Write content to file',
    params: [
      {
        name: 'path',
        type: 'string',
        required: true,
        description: 'Relative path from cwd',
      },
      {
        name: 'content',
        type: 'string',
        required: true,
        description: 'File content to write',
      },
    ],
    returnType: 'void',
    xmlBinding: {
      tagName: 'fs-write',
      attributes: ['path'],
      body: 'content',
    },
    exampleReturn: `<result tool="fs.write" status="written">
  <path>src/auth/login.ts</path>
  <bytes>428</bytes>
</result>`,
  },
  {
    name: 'fs.edit',
    description: 'Edit a file using hashline anchors. Read file with { lines: true } first to get anchors.',
    params: [
      {
        name: 'path',
        type: 'string',
        required: true,
        description: 'Relative path from cwd',
      },
      {
        name: 'edits',
        type: 'object[]',
        required: true,
        description: '{ from: string, to?: string, content?: string }[] - Array of edit operations using hashline anchors',
      },
    ],
    returnType: 'string',
    xmlBinding: {
      tagName: 'fs-edit',
      attributes: ['path'],
      children: [{ tag: 'edit-item', attributes: ['from', 'to'], body: 'content' }],
    },
    exampleReturn: `<result tool="fs.edit">
  <message>Applied 2 edit(s) to src/auth/login.ts</message>
</result>`,
  },
  {
    name: 'fs.tree',
    description: 'List directory structure with optional gitignore filtering',
    params: [
      {
        name: 'path',
        type: 'string',
        required: true,
        description: 'Relative path from cwd',
      },
      {
        name: 'options',
        type: 'string',
        required: false,
        description: '{ recursive?: boolean, maxDepth?: number, gitignore?: boolean }',
      },
    ],
    returnType: 'Array<{ path, name, type, depth }>',
    xmlBinding: {
      tagName: 'fs-tree',
      attributes: ['path'],
      selfClosing: true,
    },
    exampleReturn: `<result tool="fs.tree">
  <tree path=".">
    <dir name="src">
      <dir name="auth" />
      <dir name="utils" />
      <file name="index.ts" />
    </dir>
    <dir name="tests" />
    <file name="package.json" />
    <file name="tsconfig.json" />
  </tree>
</result>`,
  },
  {
    name: 'fs.search',
    description: 'Search file contents with regex',
    params: [
      {
        name: 'pattern',
        type: 'string',
        required: true,
        description: 'Regex pattern to search for',
      },
      {
        name: 'options',
        type: 'string',
        required: false,
        description: '{ path?: string, glob?: string }',
      },
    ],
    returnType: 'Array<{ file: string, match: string }>',
    xmlBinding: {
      tagName: 'fs-search',
      attributes: ['pattern'],
      selfClosing: true,
    },
    exampleReturn: `<result tool="fs.search">
  <match file="src/auth/login.ts" line="8">const valid = await compare(password, user.passwordHash);</match>
  <match file="src/middleware/auth.ts" line="14">if (!session?.userId) throw new Error('Unauthorized');</match>
</result>`,
  },
  {
    name: 'shell',
    description: 'Execute a shell command',
    params: [
      {
        name: 'command',
        type: 'string',
        required: true,
        description: 'Shell command to execute',
      },
    ],
    returnType: '{ stdout: string, stderr: string, exitCode: number }',
    xmlBinding: {
      tagName: 'shell',
      body: 'command',
    },
    exampleReturn: `<result tool="shell">
  <stdout>Ran 42 tests: 39 passed, 3 failed</stdout>
  <stderr></stderr>
  <exitCode>1</exitCode>
</result>`,
  },
]
