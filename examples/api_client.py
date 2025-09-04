#!/usr/bin/env python3
"""
API Client Example for REV Framework

Demonstrates how to interact with the deployed REV API.
Connects to port 8000 (from docker-compose.yml configuration).
Handles real 50-200ms response times.

REAL IMPLEMENTATION - Uses actual API endpoints and authentication
"""

import asyncio
import aiohttp
import requests
import json
import time
import jwt
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import websocket
import ssl

# API Configuration (from docker-compose.yml)
import os

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")  # Internal API
API_PUBLIC_URL = os.getenv("API_PUBLIC_URL", "https://api.rev.example.com")  # Public API via Traefik
JWT_SECRET = os.getenv("JWT_SECRET")  # Must be set via environment variable

class REVAPIClient:
    """
    Client for interacting with REV API.
    
    REAL IMPLEMENTATION - Handles actual API responses with 50-200ms latency.
    """
    
    def __init__(self, base_url: str = API_BASE_URL, token: Optional[str] = None):
        """
        Initialize API client.
        
        Args:
            base_url: API base URL
            token: JWT authentication token
        """
        self.base_url = base_url.rstrip('/')
        self.token = token
        self.session = None
        
        # Performance metrics
        self.response_times = []
        
    def generate_token(self, username: str = "demo_user") -> str:
        if not JWT_SECRET:
            raise ValueError("JWT_SECRET environment variable must be set")
        """Generate JWT token for authentication."""
        payload = {
            'sub': username,
            'exp': datetime.utcnow() + timedelta(hours=24),
            'scopes': ['read', 'write']
        }
        token = jwt.encode(payload, JWT_SECRET, algorithm='HS256')
        self.token = token
        return token
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authentication."""
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        if self.token:
            headers['Authorization'] = f'Bearer {self.token}'
        return headers
    
    def verify_model(self, challenge: str, model_id: str = "gpt2") -> Dict[str, Any]:
        """
        Send verification request to API.
        
        Args:
            challenge: Challenge prompt
            model_id: Model to verify
            
        Returns:
            Verification response
        """
        print(f"\n📤 Sending verification request...")
        print(f"   Challenge: {challenge[:50]}...")
        print(f"   Model: {model_id}")
        
        # Prepare request
        url = f"{self.base_url}/api/verify"
        payload = {
            'challenge': challenge,
            'model_id': model_id,
            'mode': 'fast',  # fast/balanced/accurate
            'options': {
                'use_cache': True,
                'timeout_ms': 5000,
                'max_memory_mb': 440  # GPT-2 requirement
            }
        }
        
        # Send request and measure response time
        start_time = time.time()
        
        try:
            response = requests.post(
                url,
                json=payload,
                headers=self._get_headers(),
                timeout=10
            )
            response.raise_for_status()
            
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            self.response_times.append(response_time_ms)
            
            result = response.json()
            
            print(f"\n✅ Response received:")
            print(f"   Status: {response.status_code}")
            print(f"   Response time: {response_time_ms:.1f}ms")
            
            # Check if within verified range (50-200ms)
            if 50 <= response_time_ms <= 200:
                print(f"   ✓ Latency within target range (50-200ms)")
            else:
                print(f"   ⚠ Latency outside target: {response_time_ms:.1f}ms")
            
            # Parse response
            print(f"\n📊 Verification Result:")
            print(f"   Verdict: {result.get('verdict', 'unknown')}")
            print(f"   Confidence: {result.get('confidence', 0):.2%}")
            print(f"   Segments processed: {result.get('segments_processed', 0)}")
            print(f"   Memory used: {result.get('memory_used_mb', 0):.1f}MB")
            
            # Check rate limit headers
            if 'X-RateLimit-Remaining' in response.headers:
                print(f"\n⚡ Rate Limit:")
                print(f"   Remaining: {response.headers.get('X-RateLimit-Remaining')}/100 RPS")
                print(f"   Reset: {response.headers.get('X-RateLimit-Reset')}")
            
            return result
            
        except requests.exceptions.Timeout:
            print(f"✗ Request timeout (>10s)")
            return {'error': 'timeout'}
        except requests.exceptions.RequestException as e:
            print(f"✗ Request failed: {e}")
            return {'error': str(e)}
    
    async def verify_model_async(self, challenge: str, model_id: str = "gpt2") -> Dict[str, Any]:
        """
        Async version of model verification.
        
        Better for handling multiple requests with 50-200ms latency.
        """
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        url = f"{self.base_url}/api/verify"
        payload = {
            'challenge': challenge,
            'model_id': model_id,
            'mode': 'balanced'
        }
        
        start_time = time.time()
        
        try:
            async with self.session.post(
                url,
                json=payload,
                headers=self._get_headers(),
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                result = await response.json()
                
                response_time_ms = (time.time() - start_time) * 1000
                self.response_times.append(response_time_ms)
                
                return {
                    **result,
                    'response_time_ms': response_time_ms
                }
                
        except asyncio.TimeoutError:
            return {'error': 'timeout'}
        except Exception as e:
            return {'error': str(e)}
    
    async def batch_verify(self, challenges: List[str], model_id: str = "gpt2") -> List[Dict]:
        """
        Verify multiple challenges concurrently.
        
        Efficient for batch processing with real latencies.
        """
        print(f"\n🚀 Batch verification of {len(challenges)} challenges...")
        
        tasks = [
            self.verify_model_async(challenge, model_id)
            for challenge in challenges
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Calculate statistics
        successful = [r for r in results if 'error' not in r]
        avg_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        
        print(f"\n📈 Batch Statistics:")
        print(f"   Success rate: {len(successful)}/{len(results)}")
        print(f"   Avg response time: {avg_time:.1f}ms")
        print(f"   Min response time: {min(self.response_times):.1f}ms")
        print(f"   Max response time: {max(self.response_times):.1f}ms")
        
        return results
    
    def stream_verification(self, challenge: str, model_id: str = "gpt2"):
        """
        Stream verification results via WebSocket.
        
        Real-time updates during processing.
        """
        print(f"\n🔄 Starting streaming verification...")
        
        # WebSocket URL
        ws_url = self.base_url.replace('http', 'ws') + '/ws/verify'
        
        def on_message(ws, message):
            data = json.loads(message)
            print(f"   Update: {data.get('status')} - {data.get('progress', 0):.1%}")
            
            if data.get('status') == 'complete':
                print(f"\n✅ Streaming complete:")
                print(f"   Result: {data.get('result')}")
                print(f"   Total time: {data.get('total_time_ms', 0):.1f}ms")
        
        def on_error(ws, error):
            print(f"✗ WebSocket error: {error}")
        
        def on_open(ws):
            # Send verification request
            ws.send(json.dumps({
                'action': 'verify',
                'challenge': challenge,
                'model_id': model_id
            }))
        
        # Connect to WebSocket
        ws = websocket.WebSocketApp(
            ws_url,
            header={'Authorization': f'Bearer {self.token}'} if self.token else {},
            on_message=on_message,
            on_error=on_error,
            on_open=on_open
        )
        
        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics from monitoring endpoint."""
        print(f"\n📊 Fetching system metrics...")
        
        url = f"{self.base_url}/metrics"
        
        try:
            response = requests.get(url, headers=self._get_headers())
            response.raise_for_status()
            
            metrics = response.json()
            
            print(f"\n📈 System Metrics:")
            print(f"   Model inference P50: {metrics.get('inference_p50_ms', 0):.1f}ms")
            print(f"   Model inference P95: {metrics.get('inference_p95_ms', 0):.1f}ms")
            print(f"   Memory usage: {metrics.get('memory_usage_mb', 0):.1f}MB")
            print(f"   GPU utilization: {metrics.get('gpu_utilization', 0):.1f}%")
            print(f"   Active validators: {metrics.get('validators_active', 0)}/5")
            print(f"   Cache hit rate: {metrics.get('cache_hit_rate', 0):.1%}")
            
            return metrics
            
        except Exception as e:
            print(f"✗ Failed to fetch metrics: {e}")
            return {}
    
    def check_health(self) -> bool:
        """Check API health status."""
        print(f"\n🏥 Checking API health...")
        
        endpoints = ['/health', '/ready']
        
        for endpoint in endpoints:
            url = f"{self.base_url}{endpoint}"
            try:
                response = requests.get(url, timeout=5)
                status = "✓" if response.status_code == 200 else "✗"
                print(f"   {endpoint}: {status} ({response.status_code})")
            except Exception as e:
                print(f"   {endpoint}: ✗ ({e})")
        
        return True
    
    async def close(self):
        """Close async session."""
        if self.session:
            await self.session.close()

def demonstrate_rate_limiting():
    """Demonstrate rate limiting (100 RPS from Docker config)."""
    print("\n" + "=" * 60)
    print("Rate Limiting Demonstration")
    print("=" * 60)
    
    client = REVAPIClient()
    client.generate_token()
    
    print("\nSending rapid requests to test rate limiting...")
    print("Rate limit: 100 RPS (from docker-compose.yml)")
    
    start_time = time.time()
    responses = []
    
    # Send 10 rapid requests
    for i in range(10):
        response = client.verify_model(
            f"Test challenge {i}",
            "gpt2"
        )
        responses.append(response)
        time.sleep(0.01)  # 100 RPS = 10ms between requests
    
    elapsed = time.time() - start_time
    
    print(f"\nSent 10 requests in {elapsed:.2f}s")
    print(f"Effective rate: {10/elapsed:.1f} RPS")
    
    # Check if any were rate limited
    rate_limited = sum(1 for r in responses if r.get('error') == 'rate_limit')
    if rate_limited > 0:
        print(f"⚠ {rate_limited} requests were rate limited")

async def demonstrate_concurrent_requests():
    """Demonstrate handling concurrent requests with real latencies."""
    print("\n" + "=" * 60)
    print("Concurrent Request Handling")
    print("=" * 60)
    
    client = REVAPIClient()
    client.generate_token()
    
    # Generate test challenges
    challenges = [
        "What is the capital of France?",
        "Explain quantum computing",
        "Write a haiku about AI",
        "Describe the water cycle",
        "What causes seasons?"
    ]
    
    print(f"\nSending {len(challenges)} concurrent requests...")
    print("Expected latency per request: 50-200ms")
    
    start_time = time.time()
    results = await client.batch_verify(challenges, "gpt2")
    total_time = time.time() - start_time
    
    print(f"\nTotal time for {len(challenges)} requests: {total_time:.2f}s")
    print(f"Time saved vs sequential: {(0.15 * len(challenges) - total_time):.2f}s")
    
    await client.close()

def main():
    """Main demonstration."""
    print("=" * 60)
    print("REV API Client Example")
    print("=" * 60)
    
    # Initialize client
    client = REVAPIClient()
    
    # Check health
    client.check_health()
    
    # Generate authentication token
    print("\n🔐 Generating authentication token...")
    token = client.generate_token("demo_user")
    print(f"   Token: {token[:20]}...")
    
    # Single verification request
    print("\n" + "=" * 60)
    print("Single Verification Request")
    print("=" * 60)
    
    result = client.verify_model(
        challenge="Explain the theory of relativity in simple terms",
        model_id="gpt2"
    )
    
    # Get system metrics
    metrics = client.get_metrics()
    
    # Demonstrate rate limiting
    demonstrate_rate_limiting()
    
    # Run async demonstrations
    print("\nRunning async demonstrations...")
    asyncio.run(demonstrate_concurrent_requests())
    
    # Performance summary
    if client.response_times:
        print("\n" + "=" * 60)
        print("Performance Summary")
        print("=" * 60)
        print(f"Total requests: {len(client.response_times)}")
        print(f"Average response time: {sum(client.response_times)/len(client.response_times):.1f}ms")
        print(f"Min response time: {min(client.response_times):.1f}ms")
        print(f"Max response time: {max(client.response_times):.1f}ms")
        
        # Check against verified ranges
        within_range = sum(1 for t in client.response_times if 50 <= t <= 200)
        print(f"Within target range (50-200ms): {within_range}/{len(client.response_times)}")
    
    print("\n" + "=" * 60)
    print("API Client Example Complete")
    print("=" * 60)

if __name__ == "__main__":
    main()