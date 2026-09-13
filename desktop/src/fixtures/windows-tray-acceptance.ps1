param([Parameter(Mandatory=$true)][string]$Evidence, [ValidateSet('Discover Models','Open Magnitude')][string]$MenuAction = 'Discover Models')
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class NativeTrayMouse {
  [DllImport("user32.dll")] public static extern bool SetCursorPos(int x, int y);
  [DllImport("user32.dll")] public static extern void mouse_event(uint flags, uint dx, uint dy, uint data, UIntPtr extra);
}
'@
function Capture([string]$Name) {
  $bounds = [System.Windows.Forms.SystemInformation]::VirtualScreen
  $image = New-Object Drawing.Bitmap($bounds.Width, $bounds.Height)
  $graphics = [Drawing.Graphics]::FromImage($image)
  try {
    $graphics.CopyFromScreen($bounds.Location, [Drawing.Point]::Empty, $bounds.Size)
    $image.Save((Join-Path $Evidence $Name), [Drawing.Imaging.ImageFormat]::Png)
  } finally { $graphics.Dispose(); $image.Dispose() }
}
function Buttons {
  $condition = New-Object Windows.Automation.PropertyCondition([Windows.Automation.AutomationElement]::ControlTypeProperty, [Windows.Automation.ControlType]::Button)
  return [Windows.Automation.AutomationElement]::RootElement.FindAll([Windows.Automation.TreeScope]::Descendants, $condition)
}
function Click([Windows.Automation.AutomationElement]$Element, [bool]$Right = $false) {
  $point = $Element.GetClickablePoint()
  [void][NativeTrayMouse]::SetCursorPos([int]$point.X, [int]$point.Y)
  [NativeTrayMouse]::mouse_event($(if ($Right) {8} else {2}), 0, 0, 0, [UIntPtr]::Zero)
  [NativeTrayMouse]::mouse_event($(if ($Right) {16} else {4}), 0, 0, 0, [UIntPtr]::Zero)
}
Capture 'windows-desktop.png'
$buttons = @(Buttons)
$buttons | ForEach-Object { "$($_.Current.Name) | $($_.Current.ClassName) | $($_.Current.AutomationId)" } | Set-Content (Join-Path $Evidence 'native-buttons.txt')
$candidates = @($buttons | Where-Object { $_.Current.Name -eq 'Magnitude' -and !$_.Current.IsOffscreen })
if ($candidates.Count -gt 1) { throw 'Multiple Magnitude buttons require native tray inspection' }
$tray = $candidates | Select-Object -First 1
if (!$tray) {
  $overflow = $buttons | Where-Object { $_.Current.Name -match '^(Show hidden icons|Notification Chevron)$' } | Select-Object -First 1
  if ($overflow) {
    Click $overflow
    Start-Sleep -Milliseconds 500
    $candidates = @(@(Buttons) | Where-Object { $_.Current.Name -eq 'Magnitude' -and !$_.Current.IsOffscreen })
    if ($candidates.Count -gt 1) { throw 'Multiple Magnitude buttons require native tray inspection' }
    $tray = $candidates | Select-Object -First 1
  }
}
if (!$tray) { throw 'Magnitude native notification-area button is not visible' }
Click $tray $true
Start-Sleep -Milliseconds 500
Capture ("windows-tray-menu-" + $MenuAction.Replace(' ','-') + '.png')
$condition = New-Object Windows.Automation.PropertyCondition([Windows.Automation.AutomationElement]::ControlTypeProperty, [Windows.Automation.ControlType]::MenuItem)
$items = @([Windows.Automation.AutomationElement]::RootElement.FindAll([Windows.Automation.TreeScope]::Descendants, $condition))
$items | ForEach-Object { $_.Current.Name } | Set-Content (Join-Path $Evidence 'native-menu-items.txt')
if (!($items | Where-Object { $_.Current.Name -eq 'Quit Magnitude' -and !$_.Current.IsOffscreen })) { throw 'Magnitude tray menu did not open' }
$action = $items | Where-Object { $_.Current.Name -eq $MenuAction -and !$_.Current.IsOffscreen } | Select-Object -First 1
if (!$action) { throw 'Requested action is missing from the native tray' }
Click $action
Write-Output "PASS actual Windows notification-area click, native menu and $MenuAction action"
