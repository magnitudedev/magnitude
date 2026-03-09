import { Sharp } from 'sharp';
import sharp from 'sharp';

export type ImageMediaType = `image/${string}`;

export class Image {
    /**
     * Wrapper for a Sharp image with conveniences for base64 conversion and manipulation
     */
    private img: Sharp;

    constructor(img: Sharp) {
        this.img = img;
    }

    static fromBase64(base64: string): Image {
        const base64Data = base64.replace(/^data:.*?;base64,/, '');
        return new Image(sharp(Buffer.from(base64Data, 'base64')));
    }

    static fromBuffer(buffer: Buffer): Image {
        return new Image(sharp(buffer));
    }

    async getFormat(): Promise<keyof sharp.FormatEnum> {
        const format = (await this.img.clone().metadata()).format;
        if (!format) throw new Error("Unable to get image format");
        return format;
    }

    async toBase64(): Promise<string> {
        const base64data = (await this.img.clone().toBuffer()).toString('base64');
        return base64data;
    }

    async toBuffer(): Promise<Buffer> {
        return await this.img.clone().toBuffer();
    }

    async saveToFile(filepath: string): Promise<void> {
        await this.img.clone().toFile(filepath);
    }

    async getDimensions(): Promise<{ width: number, height: number }> {
        const { info: { width, height } } = await this.img.clone().toBuffer({ resolveWithObject: true });
        if (!width || !height) throw new Error("Unable to get dimensions from image");
        return { width, height };
    }

    async resize(width: number, height: number): Promise<Image> {
        const resizedImage = new Image(await this.img.clone().resize({
            width: Math.round(width),
            height: Math.round(height),
            fit: 'fill',
            kernel: sharp.kernel.lanczos3
        }));
        return resizedImage;
    }
}
