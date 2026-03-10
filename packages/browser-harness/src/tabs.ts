// Keeps track of current tab, enables describing open tabs, and switching between open tabs
import logger from "./logger";
import EventEmitter from "eventemitter3";
import { BrowserContext, Page } from "playwright";
import { retryOnErrorIsSuccess } from "./common";

export interface TabEvents {
    'tabChanged': (page: Page) => void
}

export interface TabState {
    activeTab: number,
    tabs: {
        title: string,
        url: string,
    }[]
}

export interface TabManagerOptions {
    switchOnActivity?: boolean;  // Whether to automatically switch tabs when user activity is detected
}

export class TabManager {
    private context: BrowserContext;
    private activePage!: Page;
    private options: TabManagerOptions;
    private pollInterval?: NodeJS.Timeout;
    public readonly events: EventEmitter<TabEvents>;

    constructor(context: BrowserContext, options: TabManagerOptions = {}) {
        this.context = context;
        this.options = {
            switchOnActivity: true,
            ...options
        };
        this.events = new EventEmitter();

        // Track new pages
        this.context.on('page', this.onPageCreated.bind(this));
    }

    async initialize() {
        // Initialize existing pages and tracking
        await this.initializeExistingPages();
    }

    private async initializeExistingPages() {
        const pages = this.context.pages();
        logger.debug(`initializeExistingPages: found ${pages.length} existing pages`);
        
        // Set up existing pages
        for (const page of pages) {
            logger.debug(`Setting up page: ${page.url()}`);
            await this.onPageCreated(page);
        }
        
        // Start polling for activity if enabled
        if (this.options.switchOnActivity) {
            logger.debug('Starting activity polling');
            this.startActivityPolling();
        }
    }

    private async onPageCreated(page: Page) {
        logger.debug(`onPageCreated called for: ${page.url()}`);
        
        // If this is the only page or no active page set, make it active
        if (!this.activePage || this.context.pages().length === 1) {
            this.setActivePage(page);
        }

        // Only set up tracking if switchOnActivity is enabled
        if (this.options.switchOnActivity) {
            // Set up activity tracking immediately
            await this.setupPageTracking(page);

            // Re-inject tracking after navigation
            page.on('load', async () => {
                await this.setupPageTracking(page);
                
                // If this is the active page and it navigated from about:blank, re-emit tabChanged
                if (this.activePage === page && page.url() !== 'about:blank') {
                    logger.debug(`Re-emitting tabChanged after navigation to: ${page.url()}`);
                    this.events.emit('tabChanged', page);
                }
            });
            
            // Also inject on DOMContentLoaded for faster setup
            page.on('domcontentloaded', async () => {
                await this.setupPageTracking(page);
            });
        }

        // Don't automatically switch to new tabs - let the user decide
        // This prevents unwanted switches when links open in new tabs

        // Clean up when page closes
        page.on('close', () => {
            if (this.activePage === page) {
                const pages = this.context.pages();
                if (pages.length > 0) {
                    this.setActivePage(pages[0]);
                }
            }
        });
        
        // Start polling if this is the first page and switchOnActivity is enabled
        if (this.context.pages().length === 1 && this.options.switchOnActivity && !this.pollInterval) {
            logger.debug('Starting activity polling from onPageCreated (first page)');
            this.startActivityPolling();
        }
    }


    private async setupPageTracking(page: Page) {
        // Inject the tracking script with retries
        await retryOnErrorIsSuccess(
            async () => {
                    const currentUrl = page.url();
                    logger.debug(`Setting up tracking for: ${currentUrl}`);
                    
                    // Inject activity tracking
                    await page.evaluate(() => {
                    // Initialize activity tracking storage
                    if (!(window as any).__tabActivityTime) {
                        (window as any).__tabActivityTime = 0;
                    }
                    
                    // Set initial activity time to NOW since we're setting up tracking
                    // This ensures the initial page is considered active
                    (window as any).__tabActivityTime = Date.now();
                    
                    // Track any user interaction
                    const recordActivity = () => {
                        (window as any).__tabActivityTime = Date.now();
                    };
                    
                    // Remove any existing listeners to avoid duplicates
                    const listeners = (window as any).__tabListeners;
                    if (listeners) {
                        listeners.forEach((l: any) => {
                            document.removeEventListener(l.event, l.handler, true);
                            window.removeEventListener(l.event, l.handler, true);
                        });
                    }
                    
                    // Store references to our listeners
                    const newListeners: any[] = [];
                    
                    // Helper to add tracked event listener
                    const addTrackedListener = (target: any, event: string, handler: any) => {
                        target.addEventListener(event, handler, true);
                        newListeners.push({ event, handler });
                    };
                    
                    // Throttled mouse move handler
                    let lastMouseMoveTime = 0;
                    const throttledMouseMove = () => {
                        const now = Date.now();
                        if (now - lastMouseMoveTime > 500) {
                            lastMouseMoveTime = now;
                            recordActivity();
                        }
                    };
                    
                    // Mouse events
                    addTrackedListener(document, 'mousedown', recordActivity);
                    addTrackedListener(document, 'click', recordActivity);
                    addTrackedListener(document, 'mousemove', throttledMouseMove);
                    addTrackedListener(document, 'mouseenter', recordActivity);
                    
                    // Keyboard events
                    addTrackedListener(document, 'keydown', recordActivity);
                    
                    // Focus events
                    addTrackedListener(window, 'focus', recordActivity);
                    
                    // Scroll events
                    let lastScrollTime = 0;
                    const throttledScroll = () => {
                        const now = Date.now();
                        if (now - lastScrollTime > 500) {
                            lastScrollTime = now;
                            recordActivity();
                        }
                    };
                    addTrackedListener(document, 'scroll', throttledScroll);
                    
                    // Visibility change
                    addTrackedListener(document, 'visibilitychange', () => {
                        if (!document.hidden) {
                            recordActivity();
                        }
                    });
                    
                    // Store listeners for cleanup
                    (window as any).__tabListeners = newListeners;
                    
                });
            },
            { mode: 'retry_all', delayMs: 200, retryLimit: 5 }
        );
    }

    private startActivityPolling() {
        // Poll for activity every 200ms
        this.pollInterval = setInterval(async () => {
            const pages = this.context.pages();
            let mostRecentPage: Page | null = null;
            let mostRecentTime = 0;
            
            // Find the page with the most recent activity
            for (const page of pages) {
                if (page.isClosed()) continue;
                
                try {
                    const lastActivity = await page.evaluate(() => {
                        // Initialize if it doesn't exist (defensive programming)
                        if (typeof (window as any).__tabActivityTime === 'undefined') {
                            (window as any).__tabActivityTime = 0;
                            
                            // If tracking wasn't set up, return -1 as a signal
                            if (!(window as any).__tabListeners) {
                                return -1;
                            }
                        }
                        return (window as any).__tabActivityTime;
                    });
                    
                    // If tracking isn't set up (-1), inject it now
                    if (lastActivity === -1) {
                        logger.debug(`Tracking not set up for ${page.url()}, injecting now...`);
                        await this.setupPageTracking(page);
                        continue; // Skip this page for this polling cycle
                    }
                    
                    if (lastActivity > mostRecentTime) {
                        mostRecentTime = lastActivity;
                        mostRecentPage = page;
                    }
                } catch (e) {
                    // Page might be navigating
                }
            }
            
            // Switch to the most recently active page if it's different and activity is recent
            if (mostRecentPage && 
                mostRecentPage !== this.activePage && 
                mostRecentTime > Date.now() - 1000) { // Activity within last second
                logger.debug(`Activity detected on ${mostRecentPage.url()}, switching...`);
                this.setActivePage(mostRecentPage);
            }
        }, 200);
    }

    public setActivePage(page: Page) {
        const isInitialPage = !this.activePage;
        
        if (this.activePage === page && !isInitialPage) {
            return;
        }
        
        logger.debug(`Active tab changed to: ${page.url()}`);
        this.activePage = page;
        this.events.emit('tabChanged', page);

        page.removeAllListeners('framenavigated');
        page.on('framenavigated', async (frame) => {
            if (frame === page.mainFrame()) {
                // const url = frame.url();
                // await logNavigation(url);
            }
        });
    }

    async switchTab(index: number) {
        const pages = this.context.pages();
        if (index < 0 || index >= pages.length) {
            throw new Error(`Invalid tab index: ${index}`);
        }
        const page = pages[index];
        logger.debug(`Switching to tab ${index} (${page.url()})`);
        await page.bringToFront();
        this.setActivePage(page);
    }

    getActivePage() {
        return this.activePage;
    }

    getPages(): Page[] {
        return this.context.pages();
    }

    async retrieveState(): Promise<TabState> {
        let activeIndex = -1;
        let tabs = [];

        for (const [i, page] of this.context.pages().entries()) {
            if (page === this.activePage) {
                activeIndex = i;
            }

            let title: string;
            try {
                title = await page.title();
            } catch {
                logger.warn('Could not load page title while retrieving tab state');
                title = '(could not load title)';
            }

            const url = page.url();
            tabs.push({ title, url });
        }

        return {
            activeTab: activeIndex,
            tabs: tabs
        };
    }

    async debugVisibilityStates() {
        const states = await Promise.all(
            this.context.pages().map(async (page, i) => {
                try {
                    const hasFocus = await page.evaluate(() => ({
                        hasFocus: document.hasFocus(),
                        hidden: document.hidden,
                        visibilityState: document.visibilityState
                    }));
                    return { index: i, url: page.url(), ...hasFocus };
                } catch {
                    return { index: i, url: page.url(), error: true };
                }
            })
        );
        return states;
    }

    async debugFocusStates() {
        const states = await Promise.all(
            this.context.pages().map(async (page, i) => {
                try {
                    const state = await page.evaluate(() => ({
                        hasFocus: document.hasFocus(),
                        isVisible: document.visibilityState === 'visible',
                        isHidden: document.hidden,
                        url: window.location.href
                    }));
                    return { index: i, ...state };
                } catch {
                    return { index: i, url: page.url(), error: true };
                }
            })
        );
        console.log('Focus states:', states);
        return states;
    }

    // Clean up
    destroy() {
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
        }
    }
}