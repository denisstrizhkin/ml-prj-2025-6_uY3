# Simplified deployment script with essential port forwarding
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
kubectl apply -f k8s/configs/
kubectl apply -f k8s/ml/
kubectl apply -f k8s/web/

Write-Host "Waiting for pods to be ready..." -ForegroundColor Yellow

# Wait for all pods to be ready
do {
    Start-Sleep -Seconds 10
    $pods = kubectl get pods -n pinn-app -o json | ConvertFrom-Json
    $readyCount = 0
    $totalPods = $pods.items.Count

    foreach ($pod in $pods.items) {
        if ($pod.status.phase -eq "Running" -and $pod.status.containerStatuses[0].ready) {
            $readyCount++
        }
    }

    Write-Host "Ready: $readyCount/$totalPods pods" -ForegroundColor Cyan
} while ($readyCount -lt $totalPods -or $totalPods -eq 0)

Write-Host "All pods are ready!" -ForegroundColor Green

# Stop any existing port forwarding
Write-Host "Stopping existing port forwarding..." -ForegroundColor Yellow
Get-Job | Where-Object { $_.Name -like "port-forward*" } | Stop-Job | Remove-Job
Start-Sleep -Seconds 2

# Kill any existing kubectl port-forward processes
Get-Process kubectl -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

Write-Host "Starting port forwarding..." -ForegroundColor Green

# Start port forwarding with descriptive job names (FIXED PORT CONFLICTS)
$portForwards = @(
    @{Name="web-app"; LocalPort=30000; TargetPort=8000; Service="web-service"},
    @{Name="minio-api"; LocalPort=9000; TargetPort=9000; Service="minio-service"},
    @{Name="minio-console"; LocalPort=9001; TargetPort=9001; Service="minio-service"},
    @{Name="clickhouse-native"; LocalPort=19000; TargetPort=9000; Service="clickhouse-service"},  # Changed from 9000 to 19000
    @{Name="clickhouse-http"; LocalPort=18123; TargetPort=8123; Service="clickhouse-service"}     # Changed from 8123 to 18123
)

foreach ($pf in $portForwards) {
    $jobName = "port-forward-$($pf.Name)"
    Write-Host "Starting $jobName on port $($pf.LocalPort)..." -ForegroundColor Cyan

    Start-Job -Name $jobName -ScriptBlock {
        param($Namespace, $Service, $LocalPort, $TargetPort, $JobName)
        Write-Output "[$JobName] Starting port forward: localhost:${LocalPort} -> ${Service}:${TargetPort}"
        kubectl port-forward -n $Namespace service/$Service "${LocalPort}:${TargetPort}"
    } -ArgumentList "pinn-app", $pf.Service, $pf.LocalPort, $pf.TargetPort, $jobName

    Start-Sleep -Seconds 1
}

Write-Host "`nPort forwarding started!" -ForegroundColor Green
Write-Host "Access points:" -ForegroundColor Yellow
Write-Host "  Web Application:    http://localhost:30000" -ForegroundColor White
Write-Host "  MinIO Console:      http://localhost:9001 (admin/admin123)" -ForegroundColor White
Write-Host "  MinIO API:          http://localhost:9000" -ForegroundColor White
Write-Host "  ClickHouse HTTP:    http://localhost:18123 (user/123)" -ForegroundColor White    # Updated port
Write-Host "  ClickHouse Native:  localhost:19000" -ForegroundColor White                      # Updated port

Write-Host "`nUseful commands:" -ForegroundColor Cyan
#Write-Host "  Check status:        .\status.ps1" -ForegroundColor Gray
#Write-Host "  Stop port forwarding: .\stop-ports.ps1" -ForegroundColor Gray
Write-Host "  View logs:           kubectl logs -n pinn-app -l app=<service-name>" -ForegroundColor Gray

# Show job status
Write-Host "`nPort forwarding jobs:" -ForegroundColor Cyan
Get-Job | Format-Table -AutoSize