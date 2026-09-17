param([string]$WindowHandle, [int]$BlankX, [int]$MaximizeX)
Add-Type @'
using System;
using System.Runtime.InteropServices;
public class NativeChrome {
 [StructLayout(LayoutKind.Sequential)] public struct Point { public int X; public int Y; }
 [DllImport("user32.dll")] public static extern IntPtr SetThreadDpiAwarenessContext(IntPtr context);
 [DllImport("user32.dll")] public static extern bool ClientToScreen(IntPtr hwnd, ref Point point);
 [DllImport("user32.dll")] public static extern uint GetDpiForWindow(IntPtr hwnd);
 [DllImport("user32.dll")] public static extern IntPtr SendMessage(IntPtr hwnd, uint msg, IntPtr wp, IntPtr lp);
}
'@
[NativeChrome]::SetThreadDpiAwarenessContext([IntPtr]::new(-4)) | Out-Null
$hwnd = [IntPtr]::new([long]$WindowHandle)
$scale = [NativeChrome]::GetDpiForWindow($hwnd) / 96.0
function Hit([int]$x) {
 $p = New-Object NativeChrome+Point
 $p.X = [int]($x * $scale); $p.Y = [int](16 * $scale)
 [NativeChrome]::ClientToScreen($hwnd, [ref]$p) | Out-Null
 return [NativeChrome]::SendMessage($hwnd, 0x84, [IntPtr]::Zero, [IntPtr]::new(($p.Y -shl 16) -bor ($p.X -band 65535))).ToInt64()
}
@{caption = (Hit $BlankX); maximize = (Hit $MaximizeX)} | ConvertTo-Json -Compress
