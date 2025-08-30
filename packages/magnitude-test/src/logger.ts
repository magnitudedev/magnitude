import pino from 'pino';
import { PrettyOptions } from 'pino-pretty';

export const logger = pino({
    level: process.env.MAGNITUDE_LOG_LEVEL || 'warn',
    transport: process.stdout.isTTY ? {
        target: 'pino-pretty',
        options: {
            colorize: !process.env.NO_COLOR,
            translateTime: 'SYS:HH:MM:ss.l',
            ignore: 'pid,hostname',
            singleLine: true
        } satisfies PrettyOptions
    } : undefined
}).child({
    name: "runner"
});

export default logger;
