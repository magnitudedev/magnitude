import pino, { Logger as PinoLogger } from 'pino';

// Re-export Logger type for convenience
export type Logger = PinoLogger;

// Create default logger instance
const logger = pino({
    name: 'magnitude-harness',
    level: process.env.LOG_LEVEL || 'info',
    transport: process.env.NODE_ENV !== 'production' ? {
        target: 'pino-pretty',
        options: {
            colorize: true,
            ignore: 'pid,hostname',
            translateTime: 'HH:MM:ss.l'
        }
    } : undefined
});

export default logger;
