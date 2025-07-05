import { parentPort } from 'node:worker_threads';
import { pathToFileURL } from 'node:url';

if (!parentPort) {
    throw new Error('Do not use this module on the main thread');
}

parentPort.on('message', async ({ configPath }) => {
    try {
        const url = pathToFileURL(configPath).href;
        const mod = await import(url);
        // not sure why jiti/register nests in another default
        const config = mod.default?.default ?? mod.default ?? mod;
        parentPort?.postMessage({ success: true, config });
    } catch (error) {
        parentPort?.postMessage({ success: false, error });
    }
});
