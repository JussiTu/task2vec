<?php
header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: POST, OPTIONS');
header('Access-Control-Allow-Headers: Content-Type');
if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') { http_response_code(204); exit; }
if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    http_response_code(405);
    echo json_encode(['error' => 'POST required']);
    exit;
}
$body = file_get_contents('php://input');
if (!$body) { http_response_code(400); echo json_encode(['error' => 'No ticket text provided']); exit; }
$ch = curl_init('http://127.0.0.1:5001/api/analyze');
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, $body);
curl_setopt($ch, CURLOPT_HTTPHEADER, ['Content-Type: application/json', 'Expect:']);
curl_setopt($ch, CURLOPT_TIMEOUT, 90);
$r    = curl_exec($ch);
$err  = curl_error($ch);
$code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
curl_close($ch);
if ($err) { http_response_code(502); echo json_encode(['error' => $err]); exit; }
http_response_code($code);
echo $r;
