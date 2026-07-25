<#
.SYNOPSIS
    Builds and packages a Dictata release for Windows x86_64.

.DESCRIPTION
    Runs the test suite, builds the release binary, and produces a versioned
    zip in dist\ together with its SHA-256 checksum.

    Two variants can be produced:
      * gpu (default features) -> requires the Vulkan SDK at build time
      * cpu (--no-default-features) -> no SDK required, slower transcription

    The target directory is read from `cargo metadata`, so a local
    .cargo\config.toml redirecting the build (MAX_PATH workaround) is honoured.

    Nothing is committed, tagged or published: this script only produces files.

.PARAMETER Variant
    gpu (default), cpu, or both.

.PARAMETER OutDir
    Where to write the packages. Default: dist

.PARAMETER SkipTests
    Skip `cargo test` (not recommended for an actual release).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\release-windows.ps1 -Variant both
#>
[CmdletBinding()]
param(
    [ValidateSet('gpu', 'cpu', 'both')]
    [string]$Variant = 'gpu',
    [string]$OutDir = 'dist',
    [switch]$SkipTests
)

$ErrorActionPreference = 'Stop'

# Repo root = parent of scripts\
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Fail($msg) { Write-Host "!!! $msg" -ForegroundColor Red; exit 1 }

# ---------------------------------------------------------------- metadata
Step 'Reading cargo metadata'
$metaJson = & cargo metadata --format-version 1 --no-deps
if ($LASTEXITCODE -ne 0) { Fail 'cargo metadata failed' }
$meta = $metaJson | ConvertFrom-Json
$pkg = $meta.packages | Where-Object { $_.name -eq 'dictata' } | Select-Object -First 1
if ($null -eq $pkg) { Fail 'package "dictata" not found in metadata' }
$version = $pkg.version
$targetDir = $meta.target_directory
Write-Host "    dictata $version"
Write-Host "    target-dir: $targetDir"

# --------------------------------------------------------------- toolchain
Step 'Toolchain'
& rustc --version
& cargo --version

# ------------------------------------------------------------------- tests
if (-not $SkipTests) {
    Step 'cargo test'
    & cargo test
    if ($LASTEXITCODE -ne 0) { Fail 'tests failed - release aborted' }
} else {
    Write-Host '    (tests skipped)' -ForegroundColor Yellow
}

# ------------------------------------------------------------------ output
$outPath = Join-Path $root $OutDir
if (-not (Test-Path $outPath)) { New-Item -ItemType Directory -Path $outPath | Out-Null }

$variants = @()
if ($Variant -eq 'both') { $variants = @('gpu', 'cpu') } else { $variants = @($Variant) }

foreach ($v in $variants) {
    Step "Building release ($v)"
    if ($v -eq 'cpu') {
        & cargo build --release --no-default-features
    } else {
        & cargo build --release
    }
    if ($LASTEXITCODE -ne 0) { Fail "build failed ($v)" }

    $exe = Join-Path $targetDir 'release\dictata.exe'
    if (-not (Test-Path $exe)) { Fail "binary not found: $exe" }

    # build.rs only warns when rc.exe is missing, so that a contributor without
    # the resource compiler can still build and run the app. A release must not
    # ship that way: assert here. The icon and the strings come from the same
    # resource, so an empty ProductName means neither was embedded.
    $info = (Get-Item $exe).VersionInfo
    if ([string]::IsNullOrWhiteSpace($info.ProductName)) {
        Fail "no version resource in $exe - build.rs could not run rc.exe (icon and metadata missing)"
    }
    if ($info.FileVersion -ne $version) {
        Fail "stale version resource in $exe : FileVersion='$($info.FileVersion)', expected '$version'"
    }
    Write-Host "    metadata: $($info.ProductName) $($info.FileVersion), icon embedded"

    $suffix = ''
    if ($v -eq 'cpu') { $suffix = '-cpu' }
    $name = "dictata-$version-windows-x86_64$suffix"
    $stage = Join-Path $outPath $name

    # Rebuild the staging directory from scratch so a stale file is never shipped.
    if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
    New-Item -ItemType Directory -Path $stage | Out-Null

    Copy-Item $exe (Join-Path $stage 'dictata.exe')
    Copy-Item (Join-Path $root 'README.md') $stage
    Copy-Item (Join-Path $root 'LICENSE') $stage
    if (Test-Path (Join-Path $root 'CHANGELOG.md')) {
        Copy-Item (Join-Path $root 'CHANGELOG.md') $stage
    }

    # No config.json is shipped: the app writes its own defaults on first run,
    # and a developer's config.json carries personal data (app rules, LLM
    # endpoint, vocabulary).

    $zip = Join-Path $outPath "$name.zip"
    if (Test-Path $zip) { Remove-Item -Force $zip }
    Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $zip

    $hash = (Get-FileHash -Algorithm SHA256 $zip).Hash.ToLower()
    "$hash  $name.zip" | Out-File -FilePath "$zip.sha256" -Encoding ascii

    $sizeMb = [math]::Round((Get-Item $zip).Length / 1MB, 1)
    Write-Host "    $name.zip  ($sizeMb MB)" -ForegroundColor Green
    Write-Host "    sha256: $hash"
}

Step 'Done'
Write-Host "Packages in: $outPath"
Write-Host 'Reminder: nothing was committed, tagged or uploaded.'
