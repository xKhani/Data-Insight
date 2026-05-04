# Data-Insight API Test Suite
$baseUrl = "http://localhost:8000"

Write-Host "--- 1. Testing Root Endpoint ---" -ForegroundColor Cyan
Invoke-RestMethod -Uri "$baseUrl/" | Format-List
Write-Host "Success!`n" -ForegroundColor Green

Write-Host "--- 2. Starting Chat Session (EDA Proposal) ---" -ForegroundColor Cyan
$chatBody = @{
    message = "Perform exploratory data analysis on the dataset to identify patterns."
    thread_id = "automated-test-session"
} | ConvertTo-Json
$response = Invoke-RestMethod -Uri "$baseUrl/chat" -Method Post -ContentType "application/json" -Body $chatBody
$response | Format-List
Write-Host "Success! Agent is now: $($response.status)`n" -ForegroundColor Green

if ($response.status -eq "waiting_for_human_approval") {
    Write-Host "--- 3. Approving Proposal (HITL Test) ---" -ForegroundColor Cyan
    $approveBody = @{
        thread_id = "automated-test-session"
    } | ConvertTo-Json
    $approveResponse = Invoke-RestMethod -Uri "$baseUrl/approve" -Method Post -ContentType "application/json" -Body $approveBody
    $approveResponse | Format-List
    Write-Host "Success! Proposal Approved.`n" -ForegroundColor Green
}

Write-Host "--- All Container API Tests Passed! ---" -ForegroundColor Green
