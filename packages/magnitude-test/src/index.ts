// Needed for React to work properly
process.env.NODE_ENV = process.env.NODE_ENV || 'production';
export {
    test,
    beforeAll,
    afterAll,
    beforeEach,
    afterEach,
} from '@/worker/testDeclaration';
export { type MagnitudeConfig, type WebServerConfig } from '@/discovery/types';
