<#
.SYNOPSIS
    Generates assets/dictata.ico (and an optional PNG preview).

.DESCRIPTION
    The icon is drawn programmatically rather than checked in as a binary blob
    nobody can edit: change the numbers below, re-run, and the whole set of
    sizes is regenerated consistently.

    Output is a multi-size .ico (16/32/48/64/128/256) with uncompressed 32-bit
    BGRA images. PNG-compressed .ico entries are supported by Windows Vista+
    but are handled inconsistently by resource compilers, so plain DIB entries
    are used instead — a few hundred KB in an exe that is tens of MB.

.PARAMETER Preview
    Also write a 256x256 PNG at this path, to eyeball the result.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\make-icon.ps1
#>
[CmdletBinding()]
param(
    [string]$Out = '',
    [string]$Preview = ''
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing

$root = Split-Path -Parent $PSScriptRoot
if ($Out -eq '') { $Out = Join-Path $root 'assets\dictata.ico' }

# Same blue as the tray icon (src/tray.rs) so the app looks like one product.
$blue = [System.Drawing.Color]::FromArgb(255, 0x4A, 0x90, 0xE2)
$white = [System.Drawing.Color]::FromArgb(255, 255, 255, 255)

function New-RoundedPath([single]$x, [single]$y, [single]$w, [single]$h, [single]$r) {
    $p = New-Object System.Drawing.Drawing2D.GraphicsPath
    $d = $r * 2
    $p.AddArc($x, $y, $d, $d, 180, 90)
    $p.AddArc($x + $w - $d, $y, $d, $d, 270, 90)
    $p.AddArc($x + $w - $d, $y + $h - $d, $d, $d, 0, 90)
    $p.AddArc($x, $y + $h - $d, $d, $d, 90, 90)
    $p.CloseFigure()
    return $p
}

# Draws the icon at an arbitrary size. All coordinates are expressed in a
# 256x256 design space and scaled, so every size is the same drawing.
function New-IconBitmap([int]$size) {
    $bmp = New-Object System.Drawing.Bitmap $size, $size,
        ([System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $g.Clear([System.Drawing.Color]::Transparent)

    $s = $size / 256.0
    $blueBrush = New-Object System.Drawing.SolidBrush $blue
    $whiteBrush = New-Object System.Drawing.SolidBrush $white

    # Rounded-square background, full bleed.
    $bg = New-RoundedPath 0 0 $size $size (56 * $s)
    $g.FillPath($blueBrush, $bg)
    $bg.Dispose()

    # Microphone capsule.
    $cap = New-RoundedPath (100 * $s) (52 * $s) (56 * $s) (88 * $s) (28 * $s)
    $g.FillPath($whiteBrush, $cap)
    $cap.Dispose()

    # Cradle: bottom half of a circle, drawn as an arc.
    $penW = 16 * $s
    $pen = New-Object System.Drawing.Pen $white, $penW
    $pen.StartCap = [System.Drawing.Drawing2D.LineCap]::Round
    $pen.EndCap = [System.Drawing.Drawing2D.LineCap]::Round
    $r = 54 * $s
    $cx = 128 * $s
    $cy = 126 * $s
    $g.DrawArc($pen, ($cx - $r), ($cy - $r), (2 * $r), (2 * $r), 0, 180)
    $pen.Dispose()

    # Stem from the bottom of the cradle down to the base.
    $stem = New-RoundedPath (121 * $s) (172 * $s) (14 * $s) (34 * $s) (7 * $s)
    $g.FillPath($whiteBrush, $stem)
    $stem.Dispose()

    # Base.
    $base = New-RoundedPath (92 * $s) (198 * $s) (72 * $s) (16 * $s) (8 * $s)
    $g.FillPath($whiteBrush, $base)
    $base.Dispose()

    $blueBrush.Dispose()
    $whiteBrush.Dispose()
    $g.Dispose()
    return $bmp
}

# Converts a Bitmap to one ICO image block: BITMAPINFOHEADER + bottom-up BGRA
# rows + an all-zero AND mask (Windows uses the alpha channel for 32bpp icons,
# but the mask must still be present and correctly padded).
function ConvertTo-IcoBlock([System.Drawing.Bitmap]$bmp) {
    $w = $bmp.Width
    $h = $bmp.Height
    $rect = New-Object System.Drawing.Rectangle 0, 0, $w, $h
    $data = $bmp.LockBits($rect, [System.Drawing.Imaging.ImageLockMode]::ReadOnly,
        [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
    $stride = $data.Stride
    $pixels = New-Object byte[] ($stride * $h)
    [System.Runtime.InteropServices.Marshal]::Copy($data.Scan0, $pixels, 0, $pixels.Length)
    $bmp.UnlockBits($data)

    $maskStride = [math]::Ceiling($w / 8.0)
    if ($maskStride % 4 -ne 0) { $maskStride += 4 - ($maskStride % 4) }
    $maskSize = $maskStride * $h

    $ms = New-Object System.IO.MemoryStream
    $bw = New-Object System.IO.BinaryWriter $ms

    # BITMAPINFOHEADER. Height is doubled: XOR image + AND mask.
    $bw.Write([uint32]40)
    $bw.Write([int32]$w)
    $bw.Write([int32]($h * 2))
    $bw.Write([uint16]1)
    $bw.Write([uint16]32)
    $bw.Write([uint32]0)                       # BI_RGB
    $bw.Write([uint32]($w * 4 * $h + $maskSize))
    $bw.Write([int32]0)
    $bw.Write([int32]0)
    $bw.Write([uint32]0)
    $bw.Write([uint32]0)

    # XOR image, bottom-up. LockBits gives top-down rows.
    for ($y = $h - 1; $y -ge 0; $y--) {
        $bw.Write($pixels, $y * $stride, $w * 4)
    }
    # AND mask: all zeros = "opaque", alpha decides.
    $bw.Write((New-Object byte[] $maskSize), 0, $maskSize)

    $bw.Flush()
    $bytes = $ms.ToArray()
    $bw.Dispose()
    $ms.Dispose()
    return $bytes
}

# Raw top-down RGBA for the tray icon: `tray_icon::Icon::from_rgba` wants
# exactly that, and including a plain buffer at compile time keeps an ICO
# decoder out of the application. Same drawing, same source of truth.
function ConvertTo-RgbaBuffer([System.Drawing.Bitmap]$bmp) {
    $w = $bmp.Width
    $h = $bmp.Height
    $rect = New-Object System.Drawing.Rectangle 0, 0, $w, $h
    $data = $bmp.LockBits($rect, [System.Drawing.Imaging.ImageLockMode]::ReadOnly,
        [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
    $stride = $data.Stride
    $pixels = New-Object byte[] ($stride * $h)
    [System.Runtime.InteropServices.Marshal]::Copy($data.Scan0, $pixels, 0, $pixels.Length)
    $bmp.UnlockBits($data)

    # GDI+ 32bppArgb is BGRA in memory; swap to RGBA.
    $out = New-Object byte[] ($w * $h * 4)
    for ($y = 0; $y -lt $h; $y++) {
        for ($x = 0; $x -lt $w; $x++) {
            $src = $y * $stride + $x * 4
            $dst = ($y * $w + $x) * 4
            $out[$dst] = $pixels[$src + 2]      # R
            $out[$dst + 1] = $pixels[$src + 1]  # G
            $out[$dst + 2] = $pixels[$src]      # B
            $out[$dst + 3] = $pixels[$src + 3]  # A
        }
    }
    return $out
}

$sizes = @(16, 32, 48, 64, 128, 256)
$blocks = @()
$bitmaps = @()
foreach ($size in $sizes) {
    $bmp = New-IconBitmap $size
    $bitmaps += $bmp
    $blocks += , (ConvertTo-IcoBlock $bmp)
}

$outDir = Split-Path -Parent $Out
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }

$fs = [System.IO.File]::Create($Out)
$bw = New-Object System.IO.BinaryWriter $fs

# ICONDIR
$bw.Write([uint16]0)
$bw.Write([uint16]1)                 # 1 = icon
$bw.Write([uint16]$sizes.Count)

# ICONDIRENTRY table. Offsets follow the whole table.
$offset = 6 + 16 * $sizes.Count
for ($i = 0; $i -lt $sizes.Count; $i++) {
    $size = $sizes[$i]
    $dim = $size
    if ($size -ge 256) { $dim = 0 }  # 0 means 256
    $bw.Write([byte]$dim)
    $bw.Write([byte]$dim)
    $bw.Write([byte]0)               # no colour palette
    $bw.Write([byte]0)               # reserved
    $bw.Write([uint16]1)             # planes
    $bw.Write([uint16]32)            # bits per pixel
    $bw.Write([uint32]$blocks[$i].Length)
    $bw.Write([uint32]$offset)
    $offset += $blocks[$i].Length
}
foreach ($b in $blocks) { $bw.Write($b, 0, $b.Length) }

$bw.Flush()
$bw.Dispose()
$fs.Dispose()

# Tray buffer, taken from the 32x32 rendering that is already in $bitmaps.
$trayIndex = [array]::IndexOf($sizes, 32)
$trayPath = Join-Path $outDir 'tray32.rgba'
$rgba = ConvertTo-RgbaBuffer $bitmaps[$trayIndex]
[System.IO.File]::WriteAllBytes($trayPath, $rgba)
Write-Host "wrote $trayPath ($($rgba.Length) bytes, 32x32 RGBA)"

if ($Preview -ne '') {
    $big = $bitmaps[$bitmaps.Count - 1]
    $big.Save($Preview, [System.Drawing.Imaging.ImageFormat]::Png)
    Write-Host "preview: $Preview"
}
foreach ($b in $bitmaps) { $b.Dispose() }

$len = (Get-Item $Out).Length
Write-Host "wrote $Out ($len bytes, sizes: $($sizes -join ', '))"
