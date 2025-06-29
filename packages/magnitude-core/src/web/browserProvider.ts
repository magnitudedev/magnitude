import { Browser, BrowserContext, BrowserContextOptions, chromium, LaunchOptions, CDPSession } from "playwright";
import objectHash from 'object-hash';
import crypto from 'node:crypto';
import logger from "@/logger";
import { Logger } from 'pino';

const DEFAULT_BROWSER_OPTIONS: LaunchOptions = {
    headless: false,
    args: ["--disable-gpu", "--disable-blink-features=AutomationControlled"],
};

export type BrowserOptions = { instance: Browser; contextOptions?: BrowserContextOptions; }
    | { cdp: string; contextOptions?: BrowserContextOptions; }
    | { launchOptions?: LaunchOptions; contextOptions?: BrowserContextOptions; }
    | { context: BrowserContext };

interface ActiveBrowser {
    // either a browser still being launched or already resolved and ready
    browserPromise: Promise<Browser>;
    activeContextsCount: number;
}

const DEFAULT_BROWSER_CONTEXT_OPTIONS: BrowserContextOptions = {
    viewport: { width: 1024, height: 768 },
}

export class BrowserProvider {
    private activeBrowsers: Record<string, ActiveBrowser> = {};
    private logger: Logger;

    private constructor() {
        this.logger = logger.child({ name: 'browser_provider' });
    }

    public static getInstance(): BrowserProvider {
        if (!(globalThis as any).__magnitude__) {
            (globalThis as any).__magnitude__ = {};
        }

        if (!(globalThis as any).__magnitude__.browserProvider) {
            (globalThis as any).__magnitude__.browserProvider = new BrowserProvider();
        }

        return (globalThis as any).__magnitude__.browserProvider;
    }

    private async _launchOrReuseBrowser(options: LaunchOptions): Promise<ActiveBrowser> {
        // hash options
        const hash = objectHash({
            ...options,
            logger: options.logger ? crypto.randomUUID() : '' // replace unserializable logger - use UUID to force re-instance in case different loggers provided
        });
        
        let activeBrowser: ActiveBrowser;
        if (!(hash in this.activeBrowsers)) {
            this.logger.trace("Launching new browser");
            // Launch new browser, get the PROMISE
            const launchPromise = chromium.launch({ ...DEFAULT_BROWSER_OPTIONS, ...options });

            activeBrowser = {
                browserPromise: launchPromise,
                activeContextsCount: 0
            };
            // add immediately in case others need to await the same one as well
            this.activeBrowsers[hash] = activeBrowser;

            // Wait for browser to fully start
            const browser = await launchPromise;

            browser.on('disconnected', () => {
                delete this.activeBrowsers[hash];
            });

            return activeBrowser;
        } else {
            this.logger.trace("Browser with same launch options exists, reusing");
            return this.activeBrowsers[hash];
        }
    }

    public async _createAndTrackContext(options: BrowserOptions): Promise<BrowserContext> {
        const activeBrowserEntry = await this._launchOrReuseBrowser('launchOptions' in options ? options.launchOptions! : {});
        const browser = await activeBrowserEntry.browserPromise;
        
        const contextOptions = 'contextOptions' in options ? options.contextOptions : undefined;

        const context = await browser.newContext(contextOptions);

        // Get viewport dimensions from context options or use defaults
        const viewport = contextOptions?.viewport || { width: 1024, height: 768 };
        const deviceScaleFactor = contextOptions?.deviceScaleFactor || 1;

        // Apply emulation settings to any new pages created
        context.on('page', async (page) => {
            const cdpSession = await page.context().newCDPSession(page);
            await this._applyEmulationSettings(cdpSession, viewport.width, viewport.height, deviceScaleFactor);
        });

        activeBrowserEntry.activeContextsCount++;

        context.on('close', async () => {
            activeBrowserEntry.activeContextsCount--;
            if (activeBrowserEntry.activeContextsCount <= 0 && browser.isConnected()) {
                await browser.close();
            }
        });
        return context;
    }

    public async newContext(options?: BrowserOptions): Promise<BrowserContext> {
        if (options && 'context' in options) {
            // Context directly provided, we don't need to manage it
            return options.context;
        }

        const dpr = process.env.DEVICE_PIXEL_RATIO ?
            parseInt(process.env.DEVICE_PIXEL_RATIO) :
            process.platform === 'darwin' ? 2 : 1;
        
        const contextOptions = {
            ...DEFAULT_BROWSER_CONTEXT_OPTIONS,
            deviceScaleFactor: dpr,
            ...(options && 'contextOptions' in options && options.contextOptions ? options.contextOptions : {})//options.browser?.contextOptions
        };

        options = { ...options, contextOptions };

        if (process.env.MAGNTIUDE_PLAYGROUND) {
            // this.logger.trace("MAGNITUDE_PLAYGROUND environment detected, connecting to browser via CDP");
            // Playground environment - force use CDP on 9222
            //const browser = await chromium.connectOverCDP('http://localhost:9222');
            //return browser.newContext(options?.contextOptions);
            this.logger.trace("MAGNITUDE_PLAYGROUND environment detected, applying playground launch options");
            const playgroundLaunchOptions = {
                args: [
                    '--remote-debugging-port=9222',
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu'
                ]
            };
            // Overwrite any launch options, instance, or cdp configuration with playground launch options
            // Ignore context options (?)
            options = {
                launchOptions: playgroundLaunchOptions
            };
        }
        
        if ('cdp' in options) {
            const browser = await chromium.connectOverCDP(options.cdp);
            return browser.newContext(options.contextOptions);
        } else if ('instance' in options) {
            return await options.instance.newContext(options.contextOptions);
        } else if ('launchOptions' in options) {
            this.logger.trace('Creating context with custom launch options');
            return await this._createAndTrackContext(options);
        } else {
            // contextOptions might be passed but no instance | cdp | launchOptions
            this.logger.trace('Creating context for default browser options');
            return await this._createAndTrackContext(options);
        }
    }

    private async _applyEmulationSettings(cdpSession: CDPSession, width: number, height: number, deviceScaleFactor: number) {
        await cdpSession.send('Emulation.setDeviceMetricsOverride', {
            width: width,
            height: height,
            deviceScaleFactor: deviceScaleFactor,
            mobile: false,
            screenWidth: width,
            screenHeight: height,
            positionX: 0,
            positionY: 0,
            screenOrientation: { angle: 0, type: 'portraitPrimary' }
        });
    }
}