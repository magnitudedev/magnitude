import path from 'node:path';
import fs from 'node:fs';
import { glob } from 'glob';
import { Worker } from 'node:worker_threads';
import { isDeno, isBun } from 'std-env';

export async function findProjectRoot(startDir: string = process.cwd()): Promise<string | null> {
    // Try find package.json in cwd or parent folders
    let currentDir = startDir;

    // Keep track of the root directory to avoid infinite loops
    const rootDir = path.parse(currentDir).root;

    while (currentDir !== rootDir) {
        const packagePath = path.join(currentDir, 'package.json');

        try {
            // Check if package.json exists in this directory
            await fs.promises.access(packagePath, fs.constants.F_OK);
            return currentDir; // Found it!
        } catch (error) {
            // Move up one directory
            const parentDir = path.dirname(currentDir);

            // If we haven't moved up, we're at the root
            if (parentDir === currentDir) {
                return null;
            }

            currentDir = parentDir;
        }
    }

    return null; // No package.json found
}

export async function isProjectRoot(dir: string): Promise<boolean> {
    // return whether cwd has package.json (i.e. is node project)
    const packagePath = path.join(dir, 'package.json');
    try {
        // Check if package.json exists in this directory
        await fs.promises.access(packagePath, fs.constants.F_OK);
        return true;
    } catch (error) {
        return false;
    }
}


export async function discoverTestFiles(patterns: string[], cwd: string = process.cwd()): Promise<string[]> {
    //console.log(`Searching for test files matching patterns: ${patterns.join(', ')}`);

    try {
        // Use glob once with all patterns
        // Positive patterns are included, negative (! prefixed) patterns are excluded
        const files = await glob(patterns, {
            cwd,
            dot: true,        // Ignore dot files by default
            nodir: true,       // Only return files, not directories
            absolute: true    // Return paths relative to cwd
        });

        return files.map(file => path.resolve(cwd, file));
    } catch (error) {
        console.error('Error discovering test files:', error);
        return [];
    }
}

export function findConfig(searchRoot: string): string | null {
    try {
        // Use glob to find the first magnitude.config.ts file
        // Excluding node_modules and dist directories
        const configFiles = glob.sync('**/magnitude.config.{mts,ts}', {
            cwd: searchRoot,
            ignore: ['**/node_modules/**', '**/dist/**'],
            absolute: true
        });

        return configFiles.length > 0 ? configFiles[0] : null;
    } catch (error) {
        console.error('Error finding config file:', error);
        return null;
    }
}

export function readConfig(configPath: string): Promise<any> {
    return new Promise((resolve, reject) => {
        const worker = new Worker(
            new URL('./worker/readConfig.js', import.meta.url),
            {
                env: { NODE_ENV: 'test', ...process.env, },
                execArgv: !(isBun || isDeno) ? ["--import=jiti/register"] : []
            }
        );

        worker.once('message', ({ success, config, error }) => {
            worker.terminate();
            if (success) {
                resolve(config);
            } else {
                reject(new Error(error));
            }
        });
        worker.once("error", (error) => {
            worker.terminate();
            reject(error);
        });

        worker.once("online", () => {
            worker.postMessage({ configPath });
        })
    });
}