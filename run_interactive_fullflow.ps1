Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$PythonExe = "D:\Apps\Anaconda3\envs\env_dk_cdsm\python.exe"

function Write-Section {
    param([string]$Text)
    Write-Host ""
    Write-Host "============================================================"
    Write-Host $Text
    Write-Host "============================================================"
}

function Select-Option {
    param(
        [string]$Title,
        [array]$Options
    )
    Write-Section $Title
    for ($i = 0; $i -lt $Options.Count; $i++) {
        $item = $Options[$i]
        Write-Host ("[{0}] {1}" -f ($i + 1), $item.Label)
    }
    while ($true) {
        $answer = Read-Host "Select 1-$($Options.Count)"
        $index = 0
        if ([int]::TryParse($answer, [ref]$index)) {
            if ($index -ge 1 -and $index -le $Options.Count) {
                return $Options[$index - 1]
            }
        }
        Write-Host "Invalid selection. Try again."
    }
}

function Read-Default {
    param(
        [string]$Prompt,
        [string]$Default
    )
    $answer = Read-Host "$Prompt [$Default]"
    if ([string]::IsNullOrWhiteSpace($answer)) {
        return $Default
    }
    return $answer
}

function Require-File {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required file not found: $Path"
    }
}

function Get-NewestDirectory {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return $null
    }
    $items = Get-ChildItem -LiteralPath $Path -Directory | Sort-Object LastWriteTime -Descending
    if ($items.Count -eq 0) {
        return $null
    }
    return $items[0].FullName
}

function Run-Python {
    param(
        [string]$Title,
        [string[]]$ArgsList
    )
    Write-Section $Title
    Write-Host ("Command: `"{0}`" {1}" -f $PythonExe, ($ArgsList -join " "))
    & $PythonExe @ArgsList
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE"
    }
}

function Normalize-Relative {
    param([string]$Path)
    return $Path.Replace("/", "\")
}

Write-Section "DK_CDSM interactive full-flow runner"
Require-File $PythonExe
Run-Python "Verify Python interpreter" @("-c", "import sys; print(sys.executable)")

$runType = (Select-Option "Run scale" @(
    [pscustomobject]@{ Label = "full_run: full-size experiment"; Value = "full_run" },
    [pscustomobject]@{ Label = "smoke_test: quick connectivity check"; Value = "smoke_test" }
)).Value

$device = "cuda"
$tagDefault = "interactive_" + (Get-Date -Format "yyyyMMdd_HHmmss")
$tag = Read-Default "Run tag" $tagDefault

if ($device -eq "cuda") {
    Run-Python "Check CUDA availability" @(
        "-c",
        "import torch; assert torch.cuda.is_available(), 'CUDA is not available'; print(torch.cuda.get_device_name(0))"
    )
}

$collection = Select-Option "Data collection method" @(
    [pscustomobject]@{ Label = "controlled: PD tracking data"; Value = "controlled" },
    [pscustomobject]@{ Label = "uncontrolled random: open-loop random torque data"; Value = "uncontrolled_random" },
    [pscustomobject]@{ Label = "uncontrolled passive: free response data"; Value = "uncontrolled_passive" }
)

$prediction = Select-Option "Prediction model" @(
    [pscustomobject]@{ Label = "dkac: Deep Koopman affine control"; Value = "dkac" },
    [pscustomobject]@{ Label = "dkuc: Deep Koopman unconstrained control"; Value = "dkuc" },
    [pscustomobject]@{ Label = "edmd: RBF EDMD"; Value = "edmd" },
    [pscustomobject]@{ Label = "dkn: prediction-only Deep Koopman network"; Value = "dkn" }
)
$predictionMethod = $prediction.Value

$controllerOptions = @(
    [pscustomobject]@{ Label = "mpc: DKAC Koopman MPC"; Value = "mpc" },
    [pscustomobject]@{ Label = "lqr: finite-horizon Koopman LQR"; Value = "lqr" },
    [pscustomobject]@{ Label = "kilc: continuous-DKUC KILC, requires existing artifact"; Value = "kilc" },
    [pscustomobject]@{ Label = "none: only collect data and train/evaluate prediction"; Value = "none" }
)
$controller = (Select-Option "Controller" $controllerOptions).Value

if ($predictionMethod -eq "dkn" -and $controller -ne "none") {
    Write-Host ""
    Write-Host "DKN is prediction-only in this workflow. Controller is changed to none."
    $controller = "none"
}
if ($controller -eq "mpc" -and $predictionMethod -ne "dkac") {
    Write-Host ""
    Write-Host "MPC currently expects a DKAC artifact. Prediction model is changed to dkac."
    $predictionMethod = "dkac"
}

$trajectory = "none"
if ($controller -eq "mpc") {
    $trajectory = (Select-Option "End-effector trajectory for MPC" @(
        [pscustomobject]@{ Label = "star: five-point star"; Value = "star" },
        [pscustomobject]@{ Label = "circle"; Value = "circle" }
    )).Value
} elseif ($controller -eq "lqr") {
    $trajectory = (Select-Option "Reference for LQR" @(
        [pscustomobject]@{ Label = "circle: Cartesian end-effector circle"; Value = "circle" },
        [pscustomobject]@{ Label = "joint: joint-space ramp"; Value = "joint" }
    )).Value
} elseif ($controller -eq "kilc") {
    $trajectory = "circle"
    Write-Host ""
    Write-Host "KILC currently uses Cartesian circle reference."
}

$renderAnimation = $false
if (($controller -eq "mpc") -or ($controller -eq "lqr" -and $trajectory -eq "circle")) {
    $renderChoice = Select-Option "Render MuJoCo GIF after control?" @(
        [pscustomobject]@{ Label = "yes"; Value = "yes" },
        [pscustomobject]@{ Label = "no"; Value = "no" }
    )
    $renderAnimation = ($renderChoice.Value -eq "yes")
}

$dataBase = Join-Path $Root ("traj_data\outputs\" + $runType)
$collectArgs = @()
if ($collection.Value -eq "controlled") {
    $collectArgs = @(
        ".\traj_data\collect_data_controlled.py",
        "--out_dir", ".\traj_data\outputs\$runType",
        "--tag", $tag
    )
} elseif ($collection.Value -eq "uncontrolled_random") {
    $collectArgs = @(
        ".\traj_data\collect_data_uncontrolled.py",
        "--mode", "random",
        "--out_dir", ".\traj_data\outputs\$runType",
        "--tag", $tag
    )
} else {
    $collectArgs = @(
        ".\traj_data\collect_data_uncontrolled.py",
        "--mode", "passive",
        "--out_dir", ".\traj_data\outputs\$runType",
        "--tag", $tag
    )
}
Run-Python "Stage 1/4: data collection" $collectArgs

$dataRunDir = Get-NewestDirectory $dataBase
if ($null -eq $dataRunDir) {
    throw "Cannot locate collected dataset directory under $dataBase"
}
$datasetPath = Join-Path $dataRunDir "dataset.npz"
Require-File $datasetPath
Write-Host ("Dataset: {0}" -f $datasetPath)

$predictionScript = ".\prediction\$($predictionMethod)_prediction.py"
$predictionBase = Join-Path $Root ("prediction\outputs\" + $runType + "\" + $predictionMethod)
$predictionArgs = @(
    $predictionScript,
    "--train_dataset", (Normalize-Relative $datasetPath),
    "--run_type", $runType,
    "--pred_mode", "both",
    "--tag", $tag
)
if ($predictionMethod -in @("dkac", "dkuc", "dkn")) {
    $predictionArgs += @("--device", $device)
}
Run-Python "Stage 2/4: prediction training and evaluation" $predictionArgs

$artifactDir = Get-NewestDirectory $predictionBase
if ($null -eq $artifactDir) {
    throw "Cannot locate prediction artifact under $predictionBase"
}
Write-Host ("Prediction artifact: {0}" -f $artifactDir)

$controlResultDir = ""
if ($controller -ne "none") {
    if ($controller -eq "kilc") {
        Write-Host ""
        Write-Host "KILC requires an existing continuous-DKUC artifact."
        $customKilcArtifact = Read-Host "Enter continuous-DKUC artifact_dir"
        if ([string]::IsNullOrWhiteSpace($customKilcArtifact)) {
            throw "KILC artifact_dir is required."
        }
        $artifactForControl = $customKilcArtifact
    } else {
        $artifactForControl = $artifactDir
    }

    $controlBase = Join-Path $Root ("control\outputs\" + $runType + "\" + $controller)
    if ($controller -eq "mpc") {
        $controlArgs = @(
            ".\control\mpc_control.py",
            "--run_type", $runType,
            "--device", $device,
            "--artifact_dir", (Normalize-Relative $artifactForControl),
            "--trajectory", $trajectory,
            "--tag", $tag
        )
        if ($trajectory -eq "star") {
            $controlArgs += @("--period", "20", "--num_cycles", "1", "--start_hold", "0", "--radius", "0.45", "--inner_radius_ratio", "0.382")
        }
    } elseif ($controller -eq "lqr") {
        $controlArgs = @(
            ".\control\lqr_control.py",
            "--run_type", $runType,
            "--device", $device,
            "--artifact_dir", (Normalize-Relative $artifactForControl),
            "--model", $predictionMethod,
            "--task", $trajectory,
            "--tag", $tag
        )
    } else {
        $controlArgs = @(
            ".\control\kilc_control.py",
            "--run_type", $runType,
            "--device", $device,
            "--artifact_dir", (Normalize-Relative $artifactForControl),
            "--tag", $tag
        )
    }

    Run-Python "Stage 3/4: control experiment" $controlArgs
    $controlResultDir = Get-NewestDirectory $controlBase
    if ($null -eq $controlResultDir) {
        throw "Cannot locate control result under $controlBase"
    }
    Write-Host ("Control result: {0}" -f $controlResultDir)
} else {
    Write-Section "Stage 3/4: control skipped"
}

if ($renderAnimation -and $controlResultDir) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $mediaDir = Join-Path $Root ("visualization\outputs\" + $runType + "\media\" + $stamp + "_" + $controller + "_" + $predictionMethod + "_" + $trajectory)
    $renderModel = $predictionMethod
    if ($controller -eq "mpc") {
        $renderModel = "dkac"
    }
    $renderArgs = @(
        ".\visualization\entrypoints\render_animation.py",
        "--result_dir", (Normalize-Relative $controlResultDir),
        "--models", $renderModel,
        "--trajectory", $trajectory,
        "--actual_trail_color", "red",
        "--out_dir", (Normalize-Relative $mediaDir),
        "--tag", $tag
    )
    Run-Python "Stage 4/4: MuJoCo animation rendering" $renderArgs
    Write-Host ("Media output: {0}" -f $mediaDir)
} else {
    Write-Section "Stage 4/4: animation skipped"
}

Write-Section "Run summary"
Write-Host ("Run type: {0}" -f $runType)
Write-Host ("Tag: {0}" -f $tag)
Write-Host ("Dataset: {0}" -f $datasetPath)
Write-Host ("Prediction method: {0}" -f $predictionMethod)
Write-Host ("Prediction artifact: {0}" -f $artifactDir)
Write-Host ("Controller: {0}" -f $controller)
Write-Host ("Trajectory/reference: {0}" -f $trajectory)
if ($controlResultDir) {
    Write-Host ("Control result: {0}" -f $controlResultDir)
}
Write-Host ""
Write-Host "Done."
