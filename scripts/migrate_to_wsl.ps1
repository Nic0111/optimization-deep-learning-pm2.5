[CmdletBinding()]
param(
    [string]$Source = 'C:\odl-assignment',
    [string]$Distribution = 'Ubuntu-24.04',
    [switch]$ResumeExistingCopy
)

$ErrorActionPreference = 'Stop'
$sourceRoot = (Resolve-Path -LiteralPath $Source).Path
if (-not (Test-Path -LiteralPath (Join-Path $sourceRoot 'shared'))) {
    throw "The source does not look like the ODL project: $sourceRoot"
}

$installed = @(wsl.exe --list --quiet 2>$null) -replace "`0", ''
if ($installed -notcontains $Distribution) {
    throw "$Distribution is not installed. Complete the approved WSL installation first."
}

$linuxUser = (wsl.exe --distribution $Distribution -- bash -lc 'id -un').Trim()
if (-not $linuxUser) {
    throw 'Ubuntu has not completed its first-launch user setup.'
}
$targetRoot = "\\wsl.localhost\$Distribution\home\$linuxUser\code\odl-assignment"
if (Test-Path -LiteralPath $targetRoot) {
    $existing = @(Get-ChildItem -LiteralPath $targetRoot -Force -ErrorAction Stop)
    if ($existing.Count -gt 0 -and -not $ResumeExistingCopy) {
        throw "The safe migration target already contains files: $targetRoot"
    }
} else {
    New-Item -ItemType Directory -Path $targetRoot -Force | Out-Null
}

$excludedDirectories = @(
    '.venv',
    '.tmp.driveupload',
    '__pycache__',
    '.pytest_cache',
    '.ipynb_checkpoints',
    '.ruff_cache'
)
$robocopyArguments = @(
    $sourceRoot,
    $targetRoot,
    '/E',
    '/COPY:DAT',
    '/DCOPY:DAT',
    '/R:2',
    '/W:2',
    '/NFL',
    '/NDL',
    '/NJH',
    '/NJS',
    '/XD'
) + $excludedDirectories + @('/XF', '*.pyc')

& robocopy.exe @robocopyArguments
$robocopyCode = $LASTEXITCODE
if ($robocopyCode -gt 7) {
    throw "Robocopy failed with exit code $robocopyCode. The Windows source was not modified."
}

# Robocopy can expose NTFS download-origin alternate streams as ordinary files
# on Linux. They are not assignment content, so remove only those exact
# generated metadata files inside the verified migration target.
$linuxTarget = "/home/$linuxUser/code/odl-assignment"
wsl.exe --distribution $Distribution --user root chown -R "${linuxUser}:${linuxUser}" $linuxTarget
if ($LASTEXITCODE -ne 0) {
    throw 'The project copied, but assigning it to the new Linux user failed.'
}
wsl.exe --distribution $Distribution --cd $linuxTarget find . -type f -regex '.*Zone[.]Identifier.*' -delete
if ($LASTEXITCODE -ne 0) {
    throw 'The project copied, but WSL metadata-stream cleanup failed.'
}

function Get-ProjectManifest {
    param([string]$Root)
    $rootPrefix = (Resolve-Path -LiteralPath $Root).ProviderPath.TrimEnd('\') + '\'
    $excludedFragments = $excludedDirectories | ForEach-Object { [IO.Path]::DirectorySeparatorChar + $_ + [IO.Path]::DirectorySeparatorChar }
    $manifest = @{}
    Get-ChildItem -LiteralPath $Root -File -Recurse -Force | ForEach-Object {
        if (-not $_.FullName.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Manifest file escaped the project root: $($_.FullName)"
        }
        $relative = $_.FullName.Substring($rootPrefix.Length)
        $normalised = $relative.Replace('\', '/')
        $wrapped = '/' + $normalised + '/'
        $excluded = $false
        foreach ($fragment in $excludedFragments) {
            if ($wrapped -like "*$($fragment.Replace('\', '/'))*") {
                $excluded = $true
                break
            }
        }
        if (-not $excluded -and $_.Extension -ne '.pyc') {
            $manifest[$normalised] = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash
        }
    }
    return $manifest
}

$sourceManifest = Get-ProjectManifest -Root $sourceRoot
$targetManifest = Get-ProjectManifest -Root $targetRoot
$different = @($sourceManifest.Keys | Where-Object {
    -not $targetManifest.ContainsKey($_) -or $targetManifest[$_] -ne $sourceManifest[$_]
})
$extra = @($targetManifest.Keys | Where-Object { -not $sourceManifest.ContainsKey($_) })
if ($different.Count -gt 0 -or $extra.Count -gt 0) {
    throw "Migration verification failed: $($different.Count) missing/changed and $($extra.Count) extra files."
}

wsl.exe --distribution $Distribution -- bash -lc 'chmod +x ~/code/odl-assignment/scripts/*.sh'
if ($LASTEXITCODE -ne 0) {
    throw 'Files copied successfully, but setting Linux script permissions failed.'
}

Write-Output "Verified $($sourceManifest.Count) files at $targetRoot"
Write-Output 'The C: and Google Drive copies were not modified or deleted.'
Write-Output "Open the active project from Ubuntu with: cd ~/code/odl-assignment && code ."
