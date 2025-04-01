import pino from 'pino';

export const logger = pino({
    level: process.env.MAGNITUDE_LOG_LEVEL || 'info'
}).child({
    name: "remote"
});

export default logger;
