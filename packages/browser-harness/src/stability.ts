import { Image, pixelDiff } from '@magnitudedev/image';
import { Page, Request, Response } from 'playwright';
import logger, { Logger } from './logger';

// Maximum wait time for page stability in ms
const DEFAULT_PAGE_STABILITY_TIMEOUT = 5000;
// Default minimum wait time for page load
const DEFAULT_MINIMUM_WAIT_PAGE_LOAD_TIME = 500;
// Default time to wait for network requests to finish
const DEFAULT_WAIT_FOR_NETWORK_IDLE_TIME = 500;
// Default maximum time to wait for page load
const DEFAULT_MAXIMUM_WAIT_PAGE_LOAD_TIME = 5000;

export interface ImageDiff {
    /** Difference score between 0-1, where 0 means identical */
    difference: number;
    /** Optional error message if comparison failed */
    error?: string;
}

export interface PageStabilityOptions {
    /** Threshold below which screenshots are considered similar (0-1) */
    differenceThreshold?: number;
    /** Number of consecutive stable checks required to confirm stability */
    requiredStableChecks?: number;
    /** Interval between stability checks in ms */
    checkInterval?: number;
    /** Whether to log detailed stability information */
    //debug?: boolean;
    /** Minimum time to wait before getting page state */
    minimumWaitPageLoadTime?: number;
    /** Time to wait for network requests to finish before getting page state */
    waitForNetworkIdleTime?: number;
    /** Maximum time to wait for page load before proceeding anyway */
    maximumWaitPageLoadTime?: number;
    /** Whether to disable visual stability checking */
    disableVisualStability?: boolean;
}

/**
 * Utility class to check for visual stability of a page
 */
export class PageStabilityAnalyzer {
    private page!: Page;
    private options: Required<PageStabilityOptions>;
    private lastStart: number;
    private logger: Logger;

    constructor(options: PageStabilityOptions = {}) {
        //this.page = page;
        this.options = {
            differenceThreshold: options.differenceThreshold ?? 0.01,
            requiredStableChecks: options.requiredStableChecks ?? 3,
            checkInterval: options.checkInterval ?? 100, // 200ms
            minimumWaitPageLoadTime: options.minimumWaitPageLoadTime ?? DEFAULT_MINIMUM_WAIT_PAGE_LOAD_TIME,
            waitForNetworkIdleTime: options.waitForNetworkIdleTime ?? DEFAULT_WAIT_FOR_NETWORK_IDLE_TIME,
            maximumWaitPageLoadTime: options.maximumWaitPageLoadTime ?? DEFAULT_MAXIMUM_WAIT_PAGE_LOAD_TIME,
            disableVisualStability: options.disableVisualStability ?? false
        };
        this.lastStart = Date.now();
        this.logger = logger;
    }

    setActivePage(page: Page) {
        this.page = page;
    }

    private log(message: string): void {
        this.logger.debug(`[${Date.now() - this.lastStart}ms] ${message}`);
    }

    /**
     * Compare two screenshots and return their difference score
     * @returns Difference as float between 0-1, where 0 means identical
     */
    private async compareScreenshots(screenshot1: Buffer, screenshot2: Buffer): Promise<ImageDiff> {
        try {
            const img1 = Image.fromBuffer(screenshot1);
            const img2 = Image.fromBuffer(screenshot2);

            if (img1.width !== img2.width || img1.height !== img2.height) {
                return {
                    difference: 1.0,
                    error: "Image sizes don't match"
                };
            }

            const difference = pixelDiff(img1, img2);
            return { difference };
        } catch (e) {
            return {
                difference: 1.0,
                error: `Comparison failed: ${e instanceof Error ? e.message : String(e)}`
            };
        }
    }

    /**
     * Wait for the network to become stable
     * This monitors network requests and waits until there's a period of inactivity
     * @param timeout Maximum time to wait for network stability
     */
    async waitForNetworkStability(timeout?: number): Promise<void> {
        const maxWaitTime = timeout ?? this.options.maximumWaitPageLoadTime;
        const start = Date.now();
        this.log("Checking network stability");

        const pendingRequests = new Set<Request>();
        let lastActivity = Date.now();

        // Define relevant resource types and content types
        const RELEVANT_RESOURCE_TYPES = new Set([
            'document',
            'stylesheet',
            'image',
            'font',
            'script',
            'fetch',
            'xhr',
            'iframe'
        ]);

        const RELEVANT_CONTENT_TYPES = [
            'text/html',
            'text/css',
            'application/javascript',
            'image/',
            'font/',
            'application/json'
        ];

        // Additional patterns to filter out
        const IGNORED_URL_PATTERNS = [
            // Analytics and tracking
            'analytics',
            'tracking',
            'telemetry',
            'beacon',
            'metrics',
            // Ad-related
            'doubleclick',
            'adsystem',
            'adserver',
            'advertising',
            // Social media widgets
            'facebook.com/plugins',
            'platform.twitter',
            'linkedin.com/embed',
            // Live chat and support
            'livechat',
            'zendesk',
            'intercom',
            'crisp.chat',
            'hotjar',
            // Push notifications
            'push-notifications',
            'onesignal',
            'pushwoosh',
            // Background sync/heartbeat
            'heartbeat',
            'ping',
            'alive',
            // WebRTC and streaming
            'webrtc',
            'rtmp://',
            'wss://',
            // Common CDNs for dynamic content
            'cloudfront.net',
            'fastly.net'
        ];

        // Setup request handler
        const onRequest = (request: Request) => {
            // Filter by resource type
            if (!RELEVANT_RESOURCE_TYPES.has(request.resourceType())) {
                return;
            }

            // Filter out streaming, websocket, and other real-time requests
            if (['websocket', 'media', 'eventsource', 'manifest', 'other'].includes(request.resourceType())) {
                return;
            }

            // Filter out by URL patterns
            const url = request.url().toLowerCase();
            if (IGNORED_URL_PATTERNS.some(pattern => url.includes(pattern))) {
                return;
            }

            // Filter out data URLs and blob URLs
            if (url.startsWith('data:') || url.startsWith('blob:')) {
                return;
            }

            pendingRequests.add(request);
            lastActivity = Date.now();
            this.log(`Request started: ${request.url()}`);
        };

        // Setup response handler
        const onResponse = (response: Response) => {
            const request = response.request();
            if (!pendingRequests.has(request)) {
                return;
            }

            // Filter by content type if available
            const contentType = response.headers()['content-type'] || '';

            // Skip if content type indicates streaming or real-time data
            if (['streaming', 'video', 'audio', 'webm', 'mp4', 'event-stream', 'websocket', 'protobuf']
                .some(t => contentType.includes(t))) {
                pendingRequests.delete(request);
                return;
            }

            // Only process relevant content types
            if (!RELEVANT_CONTENT_TYPES.some(ct => contentType.includes(ct))) {
                pendingRequests.delete(request);
                return;
            }

            // Skip if response is too large (likely not essential for page load)
            const contentLength = parseInt(response.headers()['content-length'] || '0', 10);
            if (contentLength > 5 * 1024 * 1024) { // 5MB
                pendingRequests.delete(request);
                return;
            }

            pendingRequests.delete(request);
            lastActivity = Date.now();
            this.log(`Request resolved: ${request.url()}`);
        };

        // Add event listeners
        this.page.on('request', onRequest);
        this.page.on('response', onResponse);

        try {
            // Wait for idle time
            const startTime = Date.now();
            while (true) {
                await new Promise(resolve => setTimeout(resolve, 100)); // Sleep 100ms
                const now = Date.now();

                if (pendingRequests.size === 0 &&
                    (now - lastActivity) >= this.options.waitForNetworkIdleTime) {
                    break;
                }

                if (now - startTime > maxWaitTime) {
                    this.log(`Network timeout after ${maxWaitTime}ms with ${pendingRequests.size} pending requests`);
                    break;
                }
            }
        } finally {
            // Clean up event listeners
            this.page.removeListener('request', onRequest);
            this.page.removeListener('response', onResponse);
        }

        const totalTime = (Date.now() - start) / 1000;
        this.log(`Network stabilized in ${totalTime.toFixed(2)}s`);
    }

    /**
     * Wait for the page to become visually stable
     * @param timeout Maximum time to wait for stability in ms
     */
    async waitForVisualStability(timeout: number = DEFAULT_PAGE_STABILITY_TIMEOUT): Promise<void> {
        const start = Date.now();

        this.log("Checking visual stability");

        try {
            let lastScreenshot = await this.page.screenshot();
            let stabilityCount = 0;
            
            
            const deadline = start + timeout;
            while (Date.now() < deadline) {
                this.log(`Waiting for ${this.options.checkInterval}`);
                await this.page.waitForTimeout(this.options.checkInterval);
                this.log(`Done waiting`);

                try {
                    this.log("Taking screenshot...");
                    const currentScreenshot = await this.page.screenshot();
                    this.log("Comparing screenshots...");
                    const diffResult = await this.compareScreenshots(lastScreenshot, currentScreenshot);

                    if (diffResult.error) {
                        this.log(`Comparison error: ${diffResult.error}`);
                        stabilityCount = 0;
                    } else {
                        this.log(`Screenshot difference: ${diffResult.difference.toFixed(4)}`);

                        if (diffResult.difference < this.options.differenceThreshold) {
                            stabilityCount++;
                            if (stabilityCount >= this.options.requiredStableChecks) {
                                this.log(`Visual stability achieved (difference: ${diffResult.difference.toFixed(4)})`);
                                return;
                            }
                        } else {
                            stabilityCount = 0;
                        }
                    }

                    lastScreenshot = currentScreenshot;
                } catch (e) {
                    this.log(`Screenshot/comparison error: ${e instanceof Error ? e.message : String(e)}`);
                    stabilityCount = 0;
                }
            }

            // If we exit the loop without returning, we timed out
            this.log("Visual stability check timed out");
        } catch (e) {
            this.log(`Visual stability check error: ${e instanceof Error ? e.message : String(e)}`);
        } finally {
            const totalTime = (Date.now() - start) / 1000;
            this.log(`Visual stability check took ${totalTime.toFixed(2)}s`);
        }
    }

    /**
     * Wait for both network and visual stability
     * @param timeout Maximum time to wait for page load
     */
    async waitForStability(timeout?: number): Promise<void> {
        const maxWaitTime = timeout ?? this.options.maximumWaitPageLoadTime;
        const startTime = Date.now();
        this.lastStart = startTime;

        // Track our deadlines
        const minWaitDeadline = startTime + this.options.minimumWaitPageLoadTime;
        const maxWaitDeadline = startTime + maxWaitTime;

        this.log(`Starting stability wait (min: ${this.options.minimumWaitPageLoadTime}ms, max: ${maxWaitTime}ms)`);

        try {
            // Wait for network stability, but don't exceed max wait time
            const remainingForNetwork = Math.max(0, maxWaitDeadline - Date.now());
            if (remainingForNetwork > 0) {
                await this.waitForNetworkStability(remainingForNetwork);
            }

            // Wait for visual stability with whatever time remains, but don't exceed max wait time
            if (!this.options.disableVisualStability) {
                const remainingForVisual = Math.max(0, maxWaitDeadline - Date.now());
                if (remainingForVisual > 0) {
                    await this.waitForVisualStability(remainingForVisual);
                }
            }

            // Ensure we've waited at least the minimum time
            const now = Date.now();
            if (now < minWaitDeadline) {
                const remainingMinWait = minWaitDeadline - now;
                this.log(`Waiting additional ${remainingMinWait}ms to meet minimum wait time`);
                await new Promise(resolve => setTimeout(resolve, remainingMinWait));
            }

            const totalTime = Date.now() - startTime;
            this.log(`Page stability wait completed in ${totalTime}ms`);

        } catch (e) {
            this.log(`Error during stability wait: ${e instanceof Error ? e.message : String(e)}`);
        }
    }
}