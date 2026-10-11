param(
    [switch]$Apply,
    [switch]$CopyRuns,
    [switch]$CopyConfigs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot 'pyproject.toml') -PathType Leaf) -or
    -not (Test-Path -LiteralPath (Join-Path $ProjectRoot 'src\models\gene_aware_mae.py') -PathType Leaf) -or
    -not (Test-Path -LiteralPath (Join-Path $ProjectRoot 'scripts\train_sgmae_v1.py') -PathType Leaf)) {
    throw "Expected SG-MAE project files not found under: $ProjectRoot. Place this script in tools/prepare_project_layout.ps1."
}

Write-Host "Project root: $ProjectRoot"
if (-not $Apply) {
    Write-Host 'DRY RUN ONLY. Use -Apply to create directories or copy files.'
}

$directories = @(
    'configs\v1', 'configs\v2',
    'scripts\v1', 'scripts\v2',
    'tests\v1', 'tests\v2', 'tests\integration',
    'runs\v1\training', 'runs\v1\evaluation', 'runs\v1\representations',
    'runs\v2\graphs', 'runs\v2\training', 'runs\v2\evaluation'
)

foreach ($relative in $directories) {
    $destination = Join-Path $ProjectRoot $relative
    if (-not (Test-Path -LiteralPath $destination)) {
        Write-Host "MKDIR  $relative"
        if ($Apply) {
            New-Item -ItemType Directory -Path $destination -Force | Out-Null
        }
    }
}

$packageFiles = @('scripts\v1\__init__.py', 'scripts\v2\__init__.py')
foreach ($relative in $packageFiles) {
    $destination = Join-Path $ProjectRoot $relative
    if (-not (Test-Path -LiteralPath $destination)) {
        Write-Host "CREATE $relative"
        if ($Apply) {
            New-Item -ItemType File -Path $destination | Out-Null
        }
    }
}

$runMap = [ordered]@{
    'runs\sgmae_v1_baseline' = 'runs\v1\training\fixed_lr'
    'runs\sgmae_v1_scheduler' = 'runs\v1\training\scheduler'
    'runs\sgmae_v1_smoke' = 'runs\v1\training\smoke'
    'runs\sgmae_v1_evaluation_shaml' = 'runs\v1\evaluation\shaml'
    'runs\sgmae_v1_four_slices' = 'runs\v1\evaluation\four_slices'
    'runs\sgmae_v1_four_slices_correlations' = 'runs\v1\evaluation\four_slices_correlations'
    'runs\sgmae_cell_embeddings_cuda_fixed' = 'runs\v1\representations\embeddings'
    'runs\sgmae_cell_representation_validation' = 'runs\v1\representations\validation'
}
$configMap = [ordered]@{
    'configs\v1_training.json' = 'configs\v1\training.json'
    'configs\v1_scheduler_150.json' = 'configs\v1\scheduler_150.json'
    'configs\v1.yaml' = 'configs\v1\v1.yaml'
}

function Copy-Safe {
    param([string]$From, [string]$To, [bool]$IsDirectory)
    $source = Join-Path $ProjectRoot $From
    $destination = Join-Path $ProjectRoot $To
    if (-not (Test-Path -LiteralPath $source)) {
        Write-Host "MISSING (skip) $From"
        return
    }
    if (Test-Path -LiteralPath $destination) {
        Write-Host "EXISTS (skip)  $To"
        return
    }
    Write-Host "COPY   $From => $To"
    if (-not $Apply) { return }
    $parent = Split-Path -Path $destination -Parent
    if (-not (Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
    if ($IsDirectory) {
        Copy-Item -LiteralPath $source -Destination $destination -Recurse -ErrorAction Stop
        $originalFiles = @(Get-ChildItem -LiteralPath $source -Recurse -File -Force)
        $copiedFiles = @(Get-ChildItem -LiteralPath $destination -Recurse -File -Force)
        if ($originalFiles.Count -ne $copiedFiles.Count) {
            throw "File count verification failed after copying $From"
        }
        foreach ($file in $originalFiles) {
            $relativeFile = $file.FullName.Substring($source.Length).TrimStart('\', '/')
            $targetFile = Join-Path $destination $relativeFile
            if (-not (Test-Path -LiteralPath $targetFile -PathType Leaf)) {
                throw "File missing in copy: $targetFile"
            }
            if ($file.Length -ne (Get-Item -LiteralPath $targetFile).Length) {
                throw "File size mismatch: $targetFile"
            }
            if ($file.Extension -eq '.pt') {
                $a = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
                $b = (Get-FileHash -LiteralPath $targetFile -Algorithm SHA256).Hash
                if ($a -ne $b) { throw "Checkpoint SHA256 mismatch: $targetFile" }
            }
        }
        Write-Host "VERIFIED $From ($($originalFiles.Count) files, .pt SHA-256 checked)"
    } else {
        Copy-Item -LiteralPath $source -Destination $destination -ErrorAction Stop
        $a = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash
        $b = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash
        if ($a -ne $b) { throw "SHA256 mismatch: $To" }
        Write-Host "VERIFIED $To (SHA-256)"
    }
}

if ($CopyConfigs) {
    foreach ($pair in $configMap.GetEnumerator()) {
        Copy-Safe -From $pair.Key -To $pair.Value -IsDirectory $false
    }
}
if ($CopyRuns) {
    foreach ($pair in $runMap.GetEnumerator()) {
        Copy-Safe -From $pair.Key -To $pair.Value -IsDirectory $true
    }
}

Write-Host ''
Write-Host 'No original files have been moved or deleted.'
Write-Host 'IMPORTANT: Do not remove legacy scripts/tests/runs until their path references are migrated and regression tests pass.'
