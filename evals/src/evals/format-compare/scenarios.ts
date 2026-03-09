import { ChatMessage } from '../../types';

export const FAKE_PROJECT_CONTEXT = `Working directory: /home/user/myapp
Shell: bash
Platform: linux

Folder structure:
src/ (~12k tok)
  auth/ (~3k tok)
    login.ts
    session.ts
    middleware.ts
  db/ (~2k tok)
    connection.ts
    migrations/
  utils/ (~1k tok)
    paginate.ts
    validate.ts
  routes/ (~4k tok)
    users.ts
    posts.ts
    ... (3 more)
  index.ts
tests/ (~5k tok)
  auth/ (~2k tok)
  utils/ (~1k tok)
  ... (2 more)
config/
  default.json
  production.json
package.json
tsconfig.json
.env.example
`;
export interface ScenarioDef {
  id: string;
  description: string;
  userMessage: string;
  acceptableTools: string[];
  requiresInspect: boolean;
}

export type ScenarioChatMessage = ChatMessage;

export const SCENARIO_DEFS: ScenarioDef[] = [
  {
    id: 'explore-structure',
    description: 'Explore project directory structure',
    userMessage: 'What is the structure of this project? Show me the top-level files and directories.',
    acceptableTools: ['fs.tree', 'shell', 'fs.search'],
    requiresInspect: false,
  },
  {
    id: 'find-bug',
    description: 'Search codebase to locate authentication code',
    userMessage:
      'Users are reporting that login is failing intermittently. Find where authentication is handled in the codebase so I can investigate.',
    acceptableTools: ['fs.search', 'fs.tree', 'fs.read', 'shell'],
    requiresInspect: true,
  },
  {
    id: 'read-before-edit',
    description: 'Read a file before making changes',
    userMessage: 'There is an off-by-one error in src/utils/paginate.ts that causes the last page to be empty. Fix it.',
    acceptableTools: ['fs.read', 'fs.search', 'fs.tree', 'shell'],
    requiresInspect: true,
  },
  {
    id: 'run-tests',
    description: 'Run test suite to check status',
    userMessage: 'Run the test suite and tell me what is passing and what is failing.',
    acceptableTools: ['shell', 'fs.tree', 'fs.search', 'fs.read'],
    requiresInspect: true,
  },
  {
    id: 'multi-explore',
    description: 'Read config and explore directory in parallel',
    userMessage:
      'Read the tsconfig.json file and also show me the directory structure of src/. I want to understand the project setup.',
    acceptableTools: ['fs.read', 'fs.tree', 'shell', 'fs.search'],
    requiresInspect: true,
  },
  {
    id: 'write-new-file',
    description: 'Create a new file with known content',
    userMessage: 'Create a .gitignore file that ignores node_modules/, dist/, .env, and *.log files.',
    acceptableTools: ['fs.write', 'fs.read', 'fs.tree', 'shell'],
    requiresInspect: false,
  },
  {
    id: 'search-codebase',
    description: 'Search for specific functionality in codebase',
    userMessage: 'I need to understand how database connections are managed in this project. Find the relevant code.',
    acceptableTools: ['fs.search', 'fs.tree', 'fs.read', 'shell'],
    requiresInspect: true,
  },
  {
    id: 'complex-debug',
    description: 'Debug a failing build by running it first',
    userMessage: 'The build is broken and I do not know why. Figure out what is wrong.',
    acceptableTools: ['shell', 'fs.tree', 'fs.read', 'fs.search'],
    requiresInspect: true,
  },
];
