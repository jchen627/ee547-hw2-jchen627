#!/bin/bash

./run.sh 8081 &
SERVER_PID=$!

echo "Waiting for server startup..."
sleep 3

echo "Testing /papers..."
curl -s http://localhost:8081/papers | python -m json.tool > /dev/null && echo "✓ OK" || echo "✗ FAIL"

echo "Testing /stats..."
curl -s http://localhost:8081/stats | python -m json.tool > /dev/null && echo "✓ OK" || echo "✗ FAIL"

echo "Testing /search..."
curl -s "http://localhost:8081/search?q=machine" | python -m json.tool > /dev/null && echo "✓ OK" || echo "✗ FAIL"

echo "Testing 404..."
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8081/invalid)
if [ "$RESPONSE" = "404" ]; then echo "✓ OK"; else echo "✗ FAIL ($RESPONSE)"; fi

kill $SERVER_PID 2>/dev/null
