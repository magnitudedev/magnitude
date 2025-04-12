import { executeCliCommand, handleError } from '../utils/cliUtils.js';

/**
 * Initialize a new Magnitude project
 * @returns MCP response
 */
export async function initializeProject(): Promise<any> {
  console.log('[Setup] Initializing Magnitude project...');
  
  try {
    // Use the Magnitude CLI directly
    const output = executeCliCommand('npx magnitude init');
    
    console.log('[Setup] Magnitude project initialized successfully');
    return {
      content: [
        {
          type: 'text',
          text: `Magnitude project initialized successfully.\n\n${output}\n\nNext steps:\n1. Install magnitude-test: npm install magnitude-test\n2. Get an API key from https://app.magnitude.run/signup\n3. Set your API key in the config file or as an environment variable\n4. Run your tests with: npx magnitude`,
        },
      ],
    };
  } catch (error) {
    return handleError('Failed to initialize project', error);
  }
}
