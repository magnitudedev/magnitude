import { FakeTool } from './tools';
import { FAKE_PROJECT_CONTEXT } from './scenarios';

export interface FormatVariant {
  id: string;
  label: string;
  buildSystemPrompt(tools: FakeTool[]): string;
}

type NormalizedParam = {
  name: string;
  type: string;
  required: boolean;
  description?: string;
};

function normalizeParams(tool: FakeTool): NormalizedParam[] {
  const t = tool as any;
  const raw = t.parameters ?? t.params ?? t.input_schema ?? t.inputSchema;

  if (Array.isArray(raw)) {
    return raw.map((p: any) => ({
      name: String(p.name ?? p.key ?? 'arg'),
      type: String(p.type ?? p.schema?.type ?? 'any'),
      required: Boolean(p.required),
      description: p.description ?? p.desc,
    }));
  }

  if (raw && typeof raw === 'object') {
    const requiredList = new Set<string>(
      Array.isArray(raw.required) ? raw.required.map((x: any) => String(x)) : []
    );
    const props = raw.properties && typeof raw.properties === 'object' ? raw.properties : raw;
    return Object.entries(props)
      .filter(([k]) => k !== 'required' && k !== 'properties' && k !== 'type')
      .map(([name, spec]: [string, any]) => ({
        name,
        type: String(spec?.type ?? 'any'),
        required: Boolean(spec?.required) || requiredList.has(name),
        description: spec?.description,
      }));
  }

  return [];
}

function toolName(tool: FakeTool): string {
  const t = tool as any;
  return String(t.name ?? t.id ?? 'tool');
}

function toolDescription(tool: FakeTool): string {
  const t = tool as any;
  return String(t.description ?? t.desc ?? 'No description provided.');
}

function returnType(tool: FakeTool): string {
  const t = tool as any;
  return String(t.returnType ?? t.returns ?? t.outputType ?? 'any');
}

function tsTypeFromParam(p: NormalizedParam): string {
  const inlineType = p.description?.trim();
  if (inlineType?.startsWith('{')) {
    if (p.type === 'object[]' || inlineType.includes('[]')) {
      const typeOnly = inlineType.split(' - ')[0].trim();
      return p.type === 'object[]' && !typeOnly.includes('[]') ? `${typeOnly}[]` : typeOnly;
    }
    return inlineType;
  }

  const map: Record<string, string> = {
    string: 'string',
    number: 'number',
    integer: 'number',
    boolean: 'boolean',
    object: 'Record<string, any>',
    array: 'any[]',
    any: 'any',
  };
  return map[p.type] ?? p.type ?? 'any';
}

function renderJsToolSignature(tool: FakeTool): string {
  const t = tool as any;
  const params = normalizeParams(tool);
  const sig = params
    .map((p) => `${p.name}${p.required ? '' : '?'}: ${tsTypeFromParam(p)}`)
    .join(', ');
  const outputType = String(t.returnType ?? returnType(tool) ?? 'any');
  return `/** ${toolDescription(tool)} */\nfunction ${toolName(tool)}(${sig}): ${outputType};`;
}

function renderXmlToolExample(tool: FakeTool): string {
  const t = tool as any;
  const desc = toolDescription(tool);
  const params = normalizeParams(tool);
  const meta = params
    .map((p) => `${p.name} (${p.required ? 'required' : 'optional'}, ${p.type})`)
    .join(', ');

  const xmlBinding = t.xmlBinding;
  if (xmlBinding && typeof xmlBinding === 'object') {
    const tagName = String(xmlBinding.tagName ?? toolName(tool).replace(/\./g, '-'));
    const attributeNames = Array.isArray(xmlBinding.attributes)
      ? xmlBinding.attributes.map((a: any) => String(a))
      : [];
    const attrs = [`id="myid"`]
      .concat(attributeNames.filter((a) => a !== 'id').map((a) => `${a}="..."`))
      .join(' ');

    const renderChild = (child: any, indent = '  '): string => {
      if (typeof child === 'string') {
        return `${indent}<${child}>...</${child}>`;
      }

      const childTag = String(child?.tag ?? child?.tagName ?? child?.name ?? 'child');
      const childAttrs = Array.isArray(child?.attributes)
        ? child.attributes.map((a: any) => `${String(a)}="..."`).join(' ')
        : '';
      const childBody = child?.body ? String(child.body) : '';
      const childIsSelfClosing = Boolean(child?.selfClosing);

      if (childIsSelfClosing) {
        return `${indent}<${childTag}${childAttrs ? ` ${childAttrs}` : ''} />`;
      }

      return `${indent}<${childTag}${childAttrs ? ` ${childAttrs}` : ''}>${childBody || '...'}</${childTag}>`;
    };

    if (xmlBinding.selfClosing) {
      return `<!-- ${desc}${meta ? `. ${meta}` : ''} -->\n<${tagName}${attrs ? ` ${attrs}` : ''} />`;
    }

    const innerLines: string[] = [];
    if (xmlBinding.body) {
      innerLines.push(`  ${String(xmlBinding.body)}`);
    }

    if (Array.isArray(xmlBinding.children)) {
      innerLines.push(...xmlBinding.children.map((c: any) => renderChild(c)));
    }

    if (innerLines.length === 0) {
      return `<!-- ${desc}${meta ? `. ${meta}` : ''} -->\n<${tagName}${attrs ? ` ${attrs}` : ''}></${tagName}>`;
    }

    return `<!-- ${desc}${meta ? `. ${meta}` : ''} -->\n<${tagName}${attrs ? ` ${attrs}` : ''}>\n${innerLines.join('\n')}\n</${tagName}>`;
  }

  const tagName = tool.name.replace(/\./g, '-');
  const scalarParams = params.filter((p) => p.type !== 'string' || !['body', 'content'].includes(p.name));
  const longTextParams = params.filter((p) => p.type === 'string' && ['body', 'content'].includes(p.name));

  const attrs = [`id="myid"`]
    .concat(scalarParams.map((p) => `${p.name}="${p.name === 'query' ? 'example' : '...'}"`))
    .join(' ');

  if (longTextParams.length === 0) {
    return `<!-- ${desc}${meta ? `. ${meta}` : ''} -->\n<${tagName} ${attrs} />`;
  }

  const inner = longTextParams
    .map((p) => `  <${p.name}>...</${p.name}>`)
    .join('\n');

  return `<!-- ${desc}${meta ? `. ${meta}` : ''} -->\n<${tagName} ${attrs}>\n${inner}\n</${tagName}>`;
}

function renderAntmlToolDescription(tool: FakeTool): string {
  const params = normalizeParams(tool);
  const paramLines =
    params.length === 0
      ? '  - (no parameters)'
      : params
          .map(
            (p) =>
              `  - ${p.name}: ${p.type}${p.required ? ' (required)' : ' (optional)'}${
                p.description ? ` — ${p.description}` : ''
              }`
          )
          .join('\n');

  return `${toolName(tool)} — ${toolDescription(tool)}\n${paramLines}`;
}

const ROLE_HEADER = 'You are a helpful assistant with access to the following tools.';

export const FORMATS: FormatVariant[] = [
  {
    id: 'js-act',
    label: 'JS-Act',
    buildSystemPrompt(tools: FakeTool[]): string {
      const toolBlock = tools.map(renderJsToolSignature).join('\n\n');
      return `${ROLE_HEADER}

Project Context:
${FAKE_PROJECT_CONTEXT}

Tools (JavaScript function signatures):
${toolBlock || '/** No tools available */'}

Response format rules:
- Respond with valid JavaScript only.
- No prose, no markdown, no explanations.
- Use \`var\` for all declarations.
- End every statement with a semicolon.
- Call tools directly as functions using the signatures above.

Well-formed example:
var tree = fs.tree('src');
`;
    },
  },
  {
    id: 'js-act-inspect',
    label: 'JS-Act + Inspect',
    buildSystemPrompt(tools: FakeTool[]): string {
      const toolBlock = tools.map(renderJsToolSignature).join('\n\n');
      return `${ROLE_HEADER}

Project Context:
${FAKE_PROJECT_CONTEXT}

Tools (JavaScript function signatures):
${toolBlock || '/** No tools available */'}

/** Inspect any value to reveal it to yourself before continuing. */
function inspect(value: any): void;

Response format rules:
- Respond with valid JavaScript only.
- No prose, no markdown, no explanations.
- Use \`var\` for all declarations.
- End every statement with a semicolon.
- Progressive disclosure: after calling tools, call \`inspect(...)\` on relevant results before taking further action.
- If you do not inspect a value, assume you cannot see its contents.

Well-formed example:
var tree = fs.tree('src');
inspect(tree);
`;
    },
  },
  {
    id: 'js-act-inspect-done',
    label: 'JS-Act + Inspect + Done',
    buildSystemPrompt(tools: FakeTool[]): string {
      const toolBlock = tools.map(renderJsToolSignature).join('\n\n');
      return `${ROLE_HEADER}

Project Context:
${FAKE_PROJECT_CONTEXT}

Tools (JavaScript function signatures):
${toolBlock || '/** No tools available */'}

/** Inspect any value to reveal it to yourself before continuing. */
function inspect(value: any): void;

/** Signal that your turn is complete. Call this as the last statement every turn. */
function done(): void;

Response format rules:
- Respond with valid JavaScript only.
- No prose, no markdown, no explanations.
- Use \`var\` for all declarations.
- End every statement with a semicolon.
- Progressive disclosure: after calling tools, call \`inspect(...)\` on relevant results before taking further action.
- If you do not inspect a value, assume you cannot see its contents.
- Call \`done()\` as the very last statement of every response.

Well-formed example:
var tree = fs.tree('src');
inspect(tree);
done();
`;
    },
  },
  {
    id: 'xml-act',
    label: 'XML-Act',
    buildSystemPrompt(tools: FakeTool[]): string {
      const toolBlock = tools.map(renderXmlToolExample).join('\n\n');
      return `${ROLE_HEADER}

Project Context:
${FAKE_PROJECT_CONTEXT}

Tools (XML call format):
${toolBlock || '<!-- No tools available -->'}

Response format rules:
- Tool calls are XML elements named after the tool.
- Prose text is allowed outside tool tags.
- Place all prose and reasoning before any tool calls. Do not write any text after your tool calls.
- Every tool call must include an \`id\` attribute.
- Scalar parameters should be XML attributes.
- Long string parameters (such as \`body\` or \`content\`) should be child text elements.

Well-formed example:
I will read the auth login file.
<read id="file1" path="src/auth/login.ts" />
`;
    },
  },
  {
    id: 'xml-act-inspect',
    label: 'XML-Act + Inspect',
    buildSystemPrompt(tools: FakeTool[]): string {
      const toolBlock = tools.map(renderXmlToolExample).join('\n\n');
      return `${ROLE_HEADER}

Project Context:
${FAKE_PROJECT_CONTEXT}

Tools (XML call format):
${toolBlock || '<!-- No tools available -->'}

Additional inspect block:
<inspect>
  <ref id="emails" xpath="//email[@subject]" />
</inspect>

Response format rules:
- Tool calls are XML elements named after the tool.
- Prose text is allowed outside tool tags.
- Place all prose and reasoning before any tool calls. Do not write any text after your tool calls and inspect block.
- Every tool call must include an \`id\` attribute.
- Scalar parameters should be attributes; long string parameters (e.g., \`body\`, \`content\`) should be child text.
- Progressive disclosure: you only see results explicitly requested in the final \`<inspect>\` block.
- The \`<inspect>\` block must appear at the end of every turn.
- Without \`<inspect>\`, you see nothing.

Well-formed example:
Reading the file, then inspecting the result.
<read id="file1" path="src/auth/login.ts" />
<inspect>
  <ref id="file1" />
</inspect>
`;
    },
  },
  {
    id: 'xml-act-actions',
    label: 'XML-Act + Actions Block',
    buildSystemPrompt(tools: FakeTool[]): string {
      const toolBlock = tools.map(renderXmlToolExample).join('\n\n');
      return `${ROLE_HEADER}

Project Context:
${FAKE_PROJECT_CONTEXT}

Tools (XML call format):
${toolBlock || '<!-- No tools available -->'}

Additional inspect block:
<inspect>
  <ref id="emails" xpath="//email[@subject]" />
</inspect>

Response format rules:
- Tool calls are XML elements named after the tool.
- Prose text is allowed outside the \`<actions>\` block.
- Place all prose and reasoning before the \`<actions>\` block. Do not write any text after the \`<actions>\` block.
- Every tool call must include an \`id\` attribute.
- Scalar parameters should be attributes; long string parameters (e.g., \`body\`, \`content\`) should be child text.
- Wrap ALL tool calls and the final \`<inspect>\` block inside a single \`<actions>...</actions>\` block.
- Tool calls must only appear inside \`<actions>\`.
- Progressive disclosure: you only see results explicitly requested in the final \`<inspect>\` block.
- The \`<inspect>\` block must be the last element inside the \`<actions>\` block every turn.
- Without \`<inspect>\`, you see nothing.

Well-formed example:
I'll search for budget emails and check the calendar.
<actions>
  <grep id="search1" pattern="auth" path="src" glob="*.ts" />
  <tree id="tree1" path="src" maxDepth="2" />
  <inspect>
    <ref id="search1" />
    <ref id="tree1" />
  </inspect>
</actions>
`;
    },
  },
  {
    id: 'xml-act-actions-think',
    label: 'XML-Act + Actions + Think',
    buildSystemPrompt(tools: FakeTool[]): string {
      const toolBlock = tools.map(renderXmlToolExample).join('\n\n');
      return `${ROLE_HEADER}

Project Context:
${FAKE_PROJECT_CONTEXT}

Tools (XML call format):
${toolBlock || '<!-- No tools available -->'}

Additional inspect block:
<inspect>
  <ref id="emails" xpath="//email[@subject]" />
</inspect>

Response format rules:
- Tool calls are XML elements named after the tool.
- Prose text is allowed outside the \`<actions>\` block.
- You may optionally begin your response with a <think>...</think> block for internal reasoning before any prose or actions.
- Place all prose and reasoning before the \`<actions>\` block. Do not write any text after the \`<actions>\` block.
- Every tool call must include an \`id\` attribute.
- Scalar parameters should be attributes; long string parameters (e.g., \`body\`, \`content\`) should be child text.
- Wrap ALL tool calls and the final \`<inspect>\` block inside a single \`<actions>...</actions>\` block.
- Tool calls must only appear inside \`<actions>\`.
- Progressive disclosure: you only see results explicitly requested in the final \`<inspect>\` block.
- The \`<inspect>\` block must be the last element inside the \`<actions>\` block every turn.
- Without \`<inspect>\`, you see nothing.

Well-formed example:
<think>The user wants to search for auth files and check the directory structure. I should use grep and tree.</think>
I'll search for auth files and check the structure.
<actions>
  <grep id="search1" pattern="auth" path="src" />
  <tree id="tree1" path="src" />
  <inspect>
    <ref id="search1" />
    <ref id="tree1" />
  </inspect>
</actions>
`;
    },
  },


  {
    id: 'antml',
    label: 'ANTML',
    buildSystemPrompt(tools: FakeTool[]): string {
      const toolBlock = tools.map(renderAntmlToolDescription).join('\n\n');
      return `${ROLE_HEADER}

Project Context:
${FAKE_PROJECT_CONTEXT}

Tools:
${toolBlock || '(No tools available)'}

Response format rules (ANTML):
- You may respond with prose and/or <thinking>...</thinking>.
- Tool calls must be wrapped in <function_calls>.
- Each call uses <invoke name="tool_name"> with nested <parameter name="...">value</parameter>.
- Use exact parameter names from the tool list.

Well-formed example:
<thinking>I should search for matching emails first.</thinking>
<function_calls>
  <invoke name="fs.read">
    <parameter name="path">src/auth/login.ts</parameter>
  </invoke>
</function_calls>
`;
    },
  },
  {
    id: 'openai-native',
    label: 'OpenAI Native',
    buildSystemPrompt(): string {
      return `You are a helpful assistant with access to tools for searching emails, managing calendar, sending emails, and working with files. Use the provided tools to help the user accomplish their tasks.

Project Context:
${FAKE_PROJECT_CONTEXT}`;
    },
  },
];

export const VARIANT_IDS = FORMATS.map((f) => f.id);