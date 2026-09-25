// Renders the brand tile onto Apple's macOS app icon grid: an 824pt tile centered on a
// 1024pt canvas, with the standard drop shadow in the margin. AppKit rasterizes the SVG.
import AppKit

let arguments = CommandLine.arguments
guard arguments.count == 3, let tile = NSImage(contentsOfFile: arguments[1]) else {
  FileHandle.standardError.write("usage: render-iconset.swift <tile.svg> <output.iconset>\n".data(using: .utf8)!)
  exit(1)
}
let output = URL(fileURLWithPath: arguments[2])
try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)

func render(_ pixels: Int, to url: URL) throws {
  let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: pixels, pixelsHigh: pixels, bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
  let scale = CGFloat(pixels) / 1024
  NSGraphicsContext.saveGraphicsState()
  NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
  NSGraphicsContext.current!.imageInterpolation = .high
  let shadow = NSShadow()
  shadow.shadowColor = NSColor.black.withAlphaComponent(0.3)
  shadow.shadowOffset = NSSize(width: 0, height: -10 * scale)
  shadow.shadowBlurRadius = 20 * scale
  shadow.set()
  tile.draw(in: NSRect(x: 100 * scale, y: 100 * scale, width: 824 * scale, height: 824 * scale))
  NSGraphicsContext.restoreGraphicsState()
  try rep.representation(using: .png, properties: [:])!.write(to: url)
}

for size in [16, 32, 128, 256, 512] {
  try render(size, to: output.appendingPathComponent("icon_\(size)x\(size).png"))
  try render(size * 2, to: output.appendingPathComponent("icon_\(size)x\(size)@2x.png"))
}
