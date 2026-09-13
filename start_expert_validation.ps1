param(
    [ValidateSet("core", "supplemental", "core-video-preview", "supplemental-video-preview")]
    [string]$Round = "core",
    [ValidateRange(1024, 65535)]
    [int]$Port = 5050
)

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$App = Join-Path $ProjectRoot "expert_review_app.py"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Nedostaje projektni .venv: $Python"
}

Push-Location $ProjectRoot
try {
    & $Python ".\audits\verify_expert_validation_readiness.py"
    if ($LASTEXITCODE -ne 0) {
        throw "Paket nije prošao proveru spremnosti. Ispravi prijavljene greške pre pregleda."
    }

    switch ($Round) {
        "core" {
            & $Python $App --host 127.0.0.1 --port $Port `
                --video-packet ".\results\expert_validation_20260913\core_video_preview" --collect-only
        }
        "supplemental" {
            & $Python $App --host 127.0.0.1 --port $Port `
                --video-packet ".\results\expert_validation_20260913\supplemental_round" --collect-only
        }
        "core-video-preview" {
            & $Python $App --host 127.0.0.1 --port $Port `
                --video-packet ".\results\expert_validation_20260913\core_video_preview" --collect-only
        }
        "supplemental-video-preview" {
            & $Python $App --host 127.0.0.1 --port $Port `
                --video-packet ".\results\expert_validation_20260913\supplemental_round" --collect-only
        }
    }
}
finally {
    Pop-Location
}
