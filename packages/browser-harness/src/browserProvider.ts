import { Browser, BrowserContext, BrowserContextOptions, chromium, LaunchOptions, CDPSession } from "playwright";
import { getBrowserExecutablePath } from './browser-setup';
import objectHash from 'object-hash';
import { createId } from '@magnitudedev/generate-id';
import logger, { Logger } from "./logger";


const DEFAULT_BROWSER_OPTIONS: LaunchOptions = {
    headless: false,
    args: ["--disable-gpu", "--disable-blink-features=AutomationControlled", "--window-size=1024,768"],
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
        this.logger = logger;
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
            logger: options.logger ? createId() : '' // replace unserializable logger - use ID to force re-instance in case different loggers provided
        });
        
        let activeBrowser: ActiveBrowser;
        if (!(hash in this.activeBrowsers)) {
            this.logger.debug({ name: 'browser_provider' }, "Launching new browser");
            // Launch new browser, get the PROMISE
            const execPath = getBrowserExecutablePath() ?? undefined
            const launchPromise = chromium.launch({ ...DEFAULT_BROWSER_OPTIONS, ...(execPath ? { executablePath: execPath } : {}), ...options });

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
            this.logger.debug({ name: 'browser_provider' }, "Browser with same launch options exists, reusing");
            return this.activeBrowsers[hash];
        }
    }

    public async _createAndTrackContext(options: BrowserOptions): Promise<BrowserContext> {
        const activeBrowserEntry = await this._launchOrReuseBrowser('launchOptions' in options ? options.launchOptions! : {});
        const browser = await activeBrowserEntry.browserPromise;
        
        const contextOptions = 'contextOptions' in options ? options.contextOptions : undefined;

        const context = await browser.newContext(contextOptions);

        this._applyEmulationToContext(context, contextOptions ?? {});

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

        if ('cdp' in options) {
            const browser = await chromium.connectOverCDP(options.cdp);
            const context = browser.contexts().length > 0
                ? browser.contexts()[0]
                : await browser.newContext(contextOptions);
            this._applyEmulationToContext(context, contextOptions);
            return context;
        } else if ('instance' in options) {
            const browser = options.instance;
            const context = browser.contexts().length > 0
                ? browser.contexts()[0]
                : await browser.newContext(contextOptions);
            this._applyEmulationToContext(context, contextOptions);
            return context;
        } else if ('launchOptions' in options) {
            this.logger.debug({ name: 'browser_provider' }, 'Creating context with custom launch options');
            return await this._createAndTrackContext(options);
        } else {
            // contextOptions might be passed but no instance | cdp | launchOptions
            this.logger.debug({ name: 'browser_provider' }, 'Creating context for default browser options');
            return await this._createAndTrackContext(options);
        }
    }

    private _applyEmulationToContext(context: BrowserContext, contextOptions: BrowserContextOptions) {
        const viewport = contextOptions.viewport || { width: 1024, height: 768 };
        const deviceScaleFactor = contextOptions.deviceScaleFactor || 1;
        context.on('page', async (page) => {
            const cdpSession = await page.context().newCDPSession(page);
            await this._applyEmulationSettings(cdpSession, viewport.width, viewport.height, deviceScaleFactor);
        });
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