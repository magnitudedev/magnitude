import { spawn, type ChildProcess } from 'node:child_process';
import { setTimeout as delay } from 'node:timers/promises';
import { WebServerConfig } from './discovery/types';

export async function isServerRunning(url: string): Promise<boolean> {
    try {
        const res = await fetch(url, { method: 'HEAD' });
        return !!res.ok || res.status >= 200; // any response indicates server
    } catch {
        return false;
    }
}

export async function startWebServer(config: WebServerConfig): Promise<ChildProcess | null> {
    const { command, url, timeout = 60_000, reuseExistingServer = false } = config;

    if (reuseExistingServer && await isServerRunning(url)) {
        return null;
    }

    const child = spawn(command, { shell: true, stdio: 'inherit' });

    const startTime = Date.now();
    while (Date.now() - startTime < timeout) {
        if (await isServerRunning(url)) {
            return child;
        }
        await delay(500);
    }

    child.kill();
    throw new Error(`Timed out waiting for web server at ${url}`);
}

export function stopWebServer(proc: ChildProcess | null | undefined): void {
    if (proc) {
        proc.kill();
    }
}
