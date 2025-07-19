import { AgentConnector } from ".";
import { ActionDefinition } from '@/actions';
import { desktopActions } from '@/actions/desktopActions';
import { Observation } from "@/memory/observation";
import { Image } from "@/memory/image";
import logger from "@/logger";
import { Logger } from 'pino';
import sharp from 'sharp';

/**
 * Generic desktop automation interface.
 * Implementations can use any desktop automation technology
 * (Lume, PyAutoGUI, Windows UI Automation, etc.)
 */
export interface DesktopInterface {
    // Mouse operations
    click(x: number, y: number): Promise<void>;
    rightClick(x: number, y: number): Promise<void>;
    doubleClick(x: number, y: number): Promise<void>;
    moveCursor(x: number, y: number): Promise<void>;
    drag(fromX: number, fromY: number, toX: number, toY: number): Promise<void>;
    scroll(x: number, y: number, direction?: 'up' | 'down', amount?: number): Promise<void>;
    
    // Keyboard operations
    type(text: string): Promise<void>;
    key(key: string): Promise<void>;
    hotkey(keys: string[]): Promise<void>;
    
    // Screen operations
    screenshot(): Promise<Buffer>;
    getScreenSize(): Promise<{ width: number; height: number }>;
    
    // Navigation (browser-specific but common in desktop automation)
    navigate?(url: string): Promise<void>;
    
    // System operations
    wait(ms: number): Promise<void>;
    
    // Optional: Window management
    getActiveWindow?(): Promise<{ title: string; app: string }>;
    getOpenWindows?(): Promise<Array<{ title: string; app: string; isActive: boolean }>>;
    focusWindow?(title: string): Promise<void>;
    
    // Optional: Application control
    openApplication?(name: string): Promise<void>;
    closeApplication?(name: string): Promise<void>;
}

export interface DesktopConnectorOptions {
    desktopInterface: DesktopInterface;
    virtualScreenDimensions?: { width: number; height: number };
    minScreenshots?: number;
}

export class DesktopConnector implements AgentConnector {
    public readonly id: string = "desktop";
    private desktopInterface: DesktopInterface;
    private options: DesktopConnectorOptions;
    private logger: Logger;
    
    constructor(options: DesktopConnectorOptions) {
        this.options = options;
        this.desktopInterface = options.desktopInterface;
        this.logger = logger.child({
            name: `connectors.${this.id}`
        });
    }
    
    async onStart(): Promise<void> {
        // Desktop interface should already be initialized by the service
        this.logger.info("Desktop connector started");
    }
    
    async onStop(): Promise<void> {
        // Cleanup handled by the interface provider
        this.logger.info("Desktop connector stopped");
    }
    
    getActionSpace(): ActionDefinition<any>[] {
        return desktopActions;
    }
    
    async collectObservations(): Promise<Observation[]> {
        const observations: Observation[] = [];
        
        // Always collect screenshot
        const screenshot = await this.desktopInterface.screenshot();
        const sharpImage = sharp(screenshot);
        const image = new Image(sharpImage);
        
        // Apply virtual screen dimensions if configured
        const transformedImage = this.options.virtualScreenDimensions
            ? await image.resize(
                this.options.virtualScreenDimensions.width,
                this.options.virtualScreenDimensions.height
              )
            : image;
        
        observations.push(
            Observation.fromConnector(
                this.id,
                transformedImage,
                { 
                    type: 'screenshot', 
                    limit: this.options.minScreenshots ?? 2,
                    dedupe: true
                }
            )
        );
        
        // Optional: Window information
        if (this.desktopInterface.getOpenWindows) {
            try {
                const windows = await this.desktopInterface.getOpenWindows();
                const windowInfo = this.formatWindowInfo(windows);
                observations.push(
                    Observation.fromConnector(
                        this.id,
                        windowInfo,
                        { type: 'window-info', limit: 1 }
                    )
                );
            } catch (error) {
                this.logger.warn('Failed to get window information', error);
            }
        }
        
        return observations;
    }
    
    private formatWindowInfo(windows: Array<{ title: string; app: string; isActive: boolean }>): string {
        let info = "Open Windows:\n";
        windows.forEach(window => {
            info += `${window.isActive ? '[ACTIVE] ' : ''}${window.title} (${window.app})\n`;
        });
        return info;
    }
    
    // Expose interface for actions to use
    getInterface(): DesktopInterface {
        return this.desktopInterface;
    }
}