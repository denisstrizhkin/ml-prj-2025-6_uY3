# Improved deployment script with better waiting and background port forwarding
Write-Host "Starting PINN Application Deployment..." -ForegroundColor Green

# Build images
Write-Host "Building Docker images..." -ForegroundColor Yellow
docker build -t web-service:latest -f dockerfiles/web/Dockerfile .
docker build -t ml-service:latest -f dockerfiles/ml/Dockerfile .

# Deploy to Kubernetes
Write-Host "Deploying to Kubernetes..." -ForegroundColor Yellow
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/clickhouse/
kubectl apply -f k8s/minio/
kubectl apply -f k8s/ml/
kubectl apply -f k8s/web/

Write-Host "Waiting for services to start..." -ForegroundColor Yellow

# Wait for pods to be ready with timeout
$timeout = 180
$startTime = Get-Date
$allReady = $false

do {
    $pods = kubectl get pods -n pinn-app -o json | ConvertFrom-Json
    $readyCount = 0
    $totalPods = $pods.items.Count

    foreach ($pod in $pods.items) {
        if ($pod.status.phase -eq "Running" -and $pod.status.containerStatuses[0].ready) {
            $readyCount++
        }
    }

    $elapsed = (Get-Date) - $startTime
    Write-Host "Ready: $readyCount/$totalPods pods ($([math]::Round($elapsed.TotalSeconds))s)" -ForegroundColor Cyan

    if ($readyCount -eq $totalPods -and $totalPods -gt 0) {
        $allReady = $true
        break
    }

    if ($elapsed.TotalSeconds -gt $timeout) {
        Write-Host "Timeout waiting for pods after $timeout seconds" -ForegroundColor Red
        break
    }

    Start-Sleep -Seconds 5
} while (-not $allReady)

if (-not $allReady) {
    Write-Host "Some pods are not ready. Check with: kubectl get pods -n pinn-app" -ForegroundColor Red
    kubectl get pods -n pinn-app
    exit 1
}

Write-Host "Deployment completed!" -ForegroundColor Green
Write-Host "Web Application: http://localhost:30000" -ForegroundColor Cyan
Write-Host "MinIO Console: http://localhost:30001" -ForegroundColor Cyan

# Start port forwarding in background jobs
Write-Host "Starting port forwarding in background..." -ForegroundColor Yellow

# Stop any existing port forwarding
Get-Job | Stop-Job | Remove-Job

# Start web service port forwarding as background job
Start-Job -ScriptBlock {
    param($Namespace, $Service, $LocalPort, $TargetPort)
    Write-Host "Starting port forwarding for $Service on port $LocalPort"
    kubectl port-forward -n $Namespace service/$Service ${LocalPort}:${TargetPort}
} -ArgumentList "pinn-app", "web-service", "30000", "8000"

# Start MinIO console port forwarding as background job
Start-Job -ScriptBlock {
    param($Namespace, $Service, $LocalPort, $TargetPort)
    Write-Host "Starting port forwarding for $Service on port $LocalPort"
    kubectl port-forward -n $Namespace service/$Service ${LocalPort}:${TargetPort}
} -ArgumentList "pinn-app", "minio-service", "30001", "9001"

# Give time for port forwarding to establish
Start-Sleep -Seconds 3

Write-Host "Port forwarding jobs started!" -ForegroundColor Green
Write-Host "Check status with: .\status.ps1" -ForegroundColor Yellow
Write-Host "Stop port forwarding with: .\stop-ports.ps1" -ForegroundColor Yellow

# Show current job status
Write-Host "`nBackground jobs status:" -ForegroundColor Cyan
Get-Job | Format-Table -AutoSize