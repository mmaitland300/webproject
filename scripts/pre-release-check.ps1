# Pre-release gate: verifies this is the canonical repo, on a named branch,
# with a clean working tree. Run before any deploy or version tag.

$ErrorActionPreference = "Stop"

$canonicalPath = "C:\dev\Cursor Projects\webproject"
$currentPath   = (Get-Location).Path

if ($currentPath -ne $canonicalPath) {
    Write-Host "FAIL: cwd is '$currentPath'" -ForegroundColor Red
    Write-Host "      Releases must run from '$canonicalPath'" -ForegroundColor Red
    exit 1
}

$branch = git symbolic-ref --short HEAD 2>$null
if (-not $branch) {
    Write-Host "FAIL: HEAD is detached (no branch)" -ForegroundColor Red
    Write-Host "      Releases require a named branch" -ForegroundColor Red
    exit 1
}

$dirty = git status --porcelain
if ($dirty) {
    Write-Host "FAIL: working tree is dirty" -ForegroundColor Red
    git status --short
    exit 1
}

$sha = git rev-parse --short HEAD

Write-Host ""
Write-Host "Pre-release check passed" -ForegroundColor Green
Write-Host "  Path:   $currentPath"
Write-Host "  Branch: $branch"
Write-Host "  SHA:    $sha"
Write-Host ""
