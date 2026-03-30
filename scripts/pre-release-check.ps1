# Pre-release gate: verifies cwd is the git repository root, on a named branch,
# with a clean working tree. Run before any deploy or version tag.

$ErrorActionPreference = "Stop"

$currentPath = (Get-Location).Path
$repoRoot = (git rev-parse --show-toplevel 2>$null)
if (-not $repoRoot) {
    Write-Host "FAIL: not inside a git repository" -ForegroundColor Red
    exit 1
}
$repoRoot = $repoRoot.Trim()

if ($currentPath -ne $repoRoot) {
    Write-Host "FAIL: cwd is '$currentPath'" -ForegroundColor Red
    Write-Host "      Run from repository root: $repoRoot" -ForegroundColor Red
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
