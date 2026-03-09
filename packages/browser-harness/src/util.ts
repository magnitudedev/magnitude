export type KeypressIndicator = '<enter>' | '<tab>';

export function parseTypeContent(content: string): (string | KeypressIndicator)[] {
    // Parse content sequence potentially with special indicators <enter> or <tab> into components
    const regex = /<enter>|<tab>/g;
    const result: string[] = [];
    let lastIndex = 0;

    content.replace(regex, (match, offset) => {
        // Grab the text before the match and trim surrounding whitespace
        const text = content.slice(lastIndex, offset).trim();
        if (text) result.push(text);

        // Add the matched special indicator
        result.push(match);
        lastIndex = offset + match.length;
        return match;
    });

    // Add any remaining text after the last match
    const remaining = content.slice(lastIndex).trim();
    if (remaining) result.push(remaining);

    return result;
}

export function renderMinimalAccessibilityTree(node: any): string {
    // Escapes newlines for safe inline printing, and trims whitespace.
    const sanitize = (text: string | number | boolean): string => {
        return String(text).trim().replace(/\n/g, '\\n');
    }

    // Inner recursive function to handle state like indentation
    function recursiveFlatten(node: any, indent: string): string {
        const isTextLike = (n: any) =>
            n && (n.role === 'StaticText' || n.role === 'text') && n.name && n.name.trim();

        let childrenOutput = '';
        if (node.children) {
            const newChildren = [];
            let currentTextBuffer = [];

            for (const child of node.children) {
                if (isTextLike(child)) {
                    currentTextBuffer.push(sanitize(child.name));
                } else {
                    if (currentTextBuffer.length > 0) {
                        newChildren.push({ role: 'coalesced-text', name: currentTextBuffer.join(' ') });
                        currentTextBuffer = [];
                    }
                    newChildren.push(child);
                }
            }
            if (currentTextBuffer.length > 0) {
                newChildren.push({ role: 'coalesced-text', name: currentTextBuffer.join(' ') });
            }

            for (const child of newChildren) {
                if (child.role === 'coalesced-text') {
                    childrenOutput += `${indent}  ${child.name}\n`;
                } else {
                    childrenOutput += recursiveFlatten(child, indent + '  ');
                }
            }
        }

        const isBoringContainer = (node.role === 'generic' || node.role === 'div') && !node.name;
        if (isBoringContainer) {
            return childrenOutput;
        }

        let line = '';
        if (node.role && !['StaticText', 'text', 'RootWebArea'].includes(node.role)) {
            line = `${indent}[${node.role}]`;
            
            if (node.name && sanitize(node.name)) {
                line += ` ${sanitize(node.name)}`;
            }
            
            if (node.value !== undefined && sanitize(node.value)) {
                line += ` (value: ${sanitize(node.value)})`;
            }

            if (node.checked !== undefined) line += ` (checked: ${node.checked})`;
            if (node.disabled) line += ` (disabled)`;
            line += '\n';
        }

        return line + childrenOutput;
    }
    
    const rawFlattened = recursiveFlatten(node, '');
    return rawFlattened.split('\n').filter(line => line.trim() !== '').join('\n');
}