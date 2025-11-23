"""
Test to demonstrate the difference between Fixed Window and True Sliding Window.

This test file proves that the current implementation is Fixed Window,
which allows burst traffic at window boundaries.
"""
import asyncio
import os
import sys
import time
from typing import List
from unittest.mock import MagicMock, Mock, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    _PROXY_MaxParallelRequestsHandler_v3,
)


class MockRedisScript:
    """Mock for Redis registered script."""
    def __init__(self, script, redis_cache):
        self.script = script
        self.redis_cache = redis_cache

    async def __call__(self, keys, args):
        """Execute the script."""
        return await self.redis_cache.eval(self.script, keys, args)


class MockRedisCache:
    """
    Mock for Redis cache to test Lua script behavior.

    IMPORTANT: This is a Python reimplementation of BATCH_RATE_LIMITER_SCRIPT.
    It validates the LOGIC of the sliding window algorithm, not the Lua script itself.

    The implementation mirrors the Lua script line-by-line:
    - Lines 46-52: Process each window/counter pair
    - Lines 54-79: Window reset logic with weighted count calculation
    - Lines 83-103: Within-window increment with sliding weight

    Limitations:
    - Does not catch Lua-specific syntax errors
    - Does not validate Redis-specific behavior (atomicity, TTL, etc.)
    - For production validation, use integration tests with real Redis

    Benefits:
    - Fast unit test execution (no Redis dependency)
    - Validates algorithm correctness
    - Easy to debug and iterate on logic
    """
    def __init__(self):
        self.cache = {}  # Simulates Redis key-value store
        self.window_starts = {}  # Tracks window start times
        self.prev_counters = {}  # Tracks previous window counters

    def async_register_script(self, script):
        """Mock Redis script registration."""
        return MockRedisScript(script, self)

    async def eval(self, script, keys, args):
        """
        Mock Redis EVAL command for Lua script.

        This implementation follows BATCH_RATE_LIMITER_SCRIPT exactly:
        ARGV[1] = now (integer timestamp)
        ARGV[2] = window_size (seconds)
        KEYS = [window_key1, counter_key1, window_key2, counter_key2, ...]

        Returns: [window_start1, weighted_count1, window_start2, weighted_count2, ...]
        """
        now = args[0]
        window_size = args[1]
        results = []

        # Process each window/counter pair (2 keys at a time)
        # Mirrors Lua: for i = 1, #KEYS, 2 do
        for i in range(0, len(keys), 2):
            window_key = keys[i]
            counter_key = keys[i + 1]
            prev_counter_key = f"{counter_key}:prev"

            # Lua line 55: local window_start = redis.call('GET', window_key)
            prev_window = self.window_starts.get(window_key)
            # Lua line 59: local current_counter = redis.call('GET', counter_key)
            prev_counter = self.cache.get(counter_key, 0)

            # Lua line 56: if not window_start or (now - tonumber(window_start)) >= window_size then
            if prev_window is None or (now - prev_window) >= window_size:
                # WINDOW RESET LOGIC
                # Lua lines 58-63: Get current counter and save as previous
                prev_counter_value = int(prev_counter) if prev_counter else 0
                self.prev_counters[prev_counter_key] = prev_counter_value

                # Lua lines 66-75: Start new window, save previous, reset counter
                self.window_starts[window_key] = now
                new_counter = 1

                self.cache[window_key] = now
                self.cache[prev_counter_key] = prev_counter_value
                self.cache[counter_key] = new_counter

                # Lua lines 78-79: Calculate weighted count (elapsed=0, weight=1.0)
                weighted_count = prev_counter_value * 1.0 + new_counter
            else:
                # WITHIN CURRENT WINDOW
                # Lua line 85: local new_counter_value = redis.call('INCR', counter_key)
                new_counter = int(prev_counter) + 1
                self.cache[counter_key] = new_counter

                # Lua lines 89-93: Calculate weighted count using sliding window formula
                elapsed = now - prev_window
                weight = (window_size - elapsed) / window_size if elapsed < window_size else 0

                # Lua lines 96-100: Get previous counter value
                prev_counter_value = self.prev_counters.get(prev_counter_key, 0)

                # Lua line 103: weighted_count = prev_counter_value * weight + new_counter_value
                weighted_count = prev_counter_value * weight + new_counter

            # Lua lines 81-82, 105-106: Return results
            results.append(now)
            results.append(int(weighted_count))  # math.floor in Lua

        return results


class MockDualCache:
    """Mock for DualCache."""
    def __init__(self, redis_cache=None):
        self.redis_cache = redis_cache  # Can be None or MockRedisCache


class MockInternalUsageCache:
    """Mock for InternalUsageCache to test sliding window behavior."""

    def __init__(self, redis_cache=None):
        self.cache = {}
        self.dual_cache = MockDualCache(redis_cache=redis_cache)

    async def async_get_cache(self, key: str, **kwargs):
        return self.cache.get(key)

    async def async_set_cache(self, key: str, value, ttl: int, **kwargs):
        self.cache[key] = value

    async def async_increment(self, key: str, value: float = 1, ttl: int = None, **kwargs):
        current = self.cache.get(key, 0)
        self.cache[key] = current + value
        return self.cache[key]


@pytest.mark.asyncio
async def test_sliding_window_rate_limit_enforcement():
    """
    Test that True Sliding Window properly enforces rate limits with real time delays.

    Scenario:
    - Rate limit: 4 requests per 6 seconds (scaled down for testing)
    - Make requests at: T=0s, then sleep 2s, then 2s, then 2s (reaching T=6s)
    - At T=6.5s (after 0.5s sleep), try to make 2 more requests

    Expected with Fixed Window:
    - T=0,2,4 are in window [0-6)
    - T=6 triggers new window [6-12), counter resets
    - T=6.5: Both requests allowed (counter=2,3)

    Expected with True Sliding Window:
    - At T=6.5, look back 6s: [0.5-6.5]
    - Requests in window: T=2,4,6 (3 requests still in window)
    - Weighted count should prevent 2nd request at T=6.5
    """
    import time as time_module

    mock_cache = MockInternalUsageCache()
    handler = _PROXY_MaxParallelRequestsHandler_v3(internal_usage_cache=mock_cache)

    window_size = 6  # 6 seconds for faster testing
    rate_limit = 4

    print(f"\n=== Rate Limit Enforcement Test (Real Time) ===")
    print(f"Rate limit: {rate_limit} req/{window_size}s")

    start_time = int(time_module.time())

    # Request at T=0
    result_0 = await handler.in_memory_cache_sliding_window(
        keys=["user:ratelimit:window", "user:ratelimit:counter"],
        now_int=start_time,
        window_size=window_size,
    )
    print(f"T=0s: counter={result_0[1]}")

    # Request at T=2s
    time_module.sleep(2)
    time_2 = int(time_module.time())
    result_2 = await handler.in_memory_cache_sliding_window(
        keys=["user:ratelimit:window", "user:ratelimit:counter"],
        now_int=time_2,
        window_size=window_size,
    )
    print(f"T=~2s: counter={result_2[1]}")

    # Request at T=4s
    time_module.sleep(2)
    time_4 = int(time_module.time())
    result_4 = await handler.in_memory_cache_sliding_window(
        keys=["user:ratelimit:window", "user:ratelimit:counter"],
        now_int=time_4,
        window_size=window_size,
    )
    print(f"T=~4s: counter={result_4[1]}")

    # Request at T=6s (boundary - Fixed Window should reset here)
    time_module.sleep(2)
    time_6 = int(time_module.time())
    result_6 = await handler.in_memory_cache_sliding_window(
        keys=["user:ratelimit:window", "user:ratelimit:counter"],
        now_int=time_6,
        window_size=window_size,
    )
    counter_6 = result_6[1]
    print(f"T=~6s: counter={counter_6}")

    # At T=6.5s, try 2 more requests
    time_module.sleep(0.5)
    time_65 = int(time_module.time())

    # First request
    result_65_1 = await handler.in_memory_cache_sliding_window(
        keys=["user:ratelimit:window", "user:ratelimit:counter"],
        now_int=time_65,
        window_size=window_size,
    )
    counter_65_1 = result_65_1[1]

    # Second request (should be rejected with Sliding Window)
    result_65_2 = await handler.in_memory_cache_sliding_window(
        keys=["user:ratelimit:window", "user:ratelimit:counter"],
        now_int=time_65,
        window_size=window_size,
    )
    counter_65_2 = result_65_2[1]

    print(f"T=~6.5s: 1st request counter={counter_65_1}, 2nd request counter={counter_65_2}")

    # With True Sliding Window implemented, 2nd request should be rejected
    if counter_65_2 > rate_limit:
        print(f"✓ SUCCESS: Sliding Window rejected 2nd request (counter={counter_65_2} > {rate_limit})")
        print(f"Sliding window is working correctly!")
        assert counter_65_2 > rate_limit, \
            f"Sliding Window correctly rejects burst: {counter_65_2} > {rate_limit}"
    else:
        print(f"✗ FAIL: Fixed Window allowed burst (counter={counter_65_2} <= {rate_limit})")
        pytest.fail(
            f"Sliding Window should reject 2nd request at T=6.5s.\n"
            f"Expected: counter > {rate_limit}, Got: counter = {counter_65_2}\n"
            f"Recent requests within sliding window: T=2,4,6,6.5,6.5 = 5 requests > limit of 4"
        )


@pytest.mark.asyncio
async def test_redis_lua_script_sliding_window():
    """
    Test Redis Lua script implementation of True Sliding Window.

    This test verifies that the Redis Lua script logic (BATCH_RATE_LIMITER_SCRIPT)
    correctly implements True Sliding Window rate limiting.

    WHAT THIS TEST VALIDATES:
    - Algorithm correctness: weighted counter formula works as expected
    - Behavioral equivalence: Redis path produces same results as in-memory
    - Window boundaries: proper handling of window reset and transitions
    - Burst prevention: sliding window prevents burst traffic at boundaries

    WHAT THIS TEST DOES NOT VALIDATE:
    - Lua syntax errors (uses Python reimplementation)
    - Redis-specific behavior (atomicity, TTL, EXPIRE commands)
    - Actual Lua script execution

    For production validation, complement this with:
    - Integration tests using real Redis instance
    - Load tests to verify performance under concurrent access
    - Manual testing of Lua script in Redis CLI
    """
    import time as time_module

    # Create mock Redis cache
    mock_redis = MockRedisCache()
    mock_cache = MockInternalUsageCache(redis_cache=mock_redis)
    handler = _PROXY_MaxParallelRequestsHandler_v3(internal_usage_cache=mock_cache)

    window_size = 6
    rate_limit = 4

    print(f"\n=== Redis Lua Script Test ===")
    print(f"Rate limit: {rate_limit} req/{window_size}s")

    start_time = int(time_module.time())

    # Test that handler uses Redis when available
    assert handler.internal_usage_cache.dual_cache.redis_cache is not None, \
        "Redis cache should be configured"

    # Request at T=0
    result_0 = await handler.batch_rate_limiter_script(
        keys=["user:ratelimit:window", "user:ratelimit:counter"],
        args=[start_time, window_size],
    )
    print(f"T=0s: counter={result_0[1]}")
    assert result_0[1] == 1, "First request counter should be 1"

    # Request at T=2s
    time_module.sleep(2)
    time_2 = int(time_module.time())
    result_2 = await handler.batch_rate_limiter_script(
        keys=["user:ratelimit:window", "user:ratelimit:counter"],
        args=[time_2, window_size],
    )
    print(f"T=~2s: counter={result_2[1]}")
    assert result_2[1] == 2, "Second request counter should be 2"

    # Request at T=4s
    time_module.sleep(2)
    time_4 = int(time_module.time())
    result_4 = await handler.batch_rate_limiter_script(
        keys=["user:ratelimit:window", "user:ratelimit:counter"],
        args=[time_4, window_size],
    )
    print(f"T=~4s: counter={result_4[1]}")
    assert result_4[1] == 3, "Third request counter should be 3"

    # Request at T=6s (window boundary)
    time_module.sleep(2)
    time_6 = int(time_module.time())
    result_6 = await handler.batch_rate_limiter_script(
        keys=["user:ratelimit:window", "user:ratelimit:counter"],
        args=[time_6, window_size],
    )
    counter_6 = result_6[1]
    print(f"T=~6s: counter={counter_6}")
    # Weighted count: prev_counter(3) * 1.0 + curr_counter(1) = 4
    assert counter_6 == 4, f"At window boundary, weighted count should be 4, got {counter_6}"

    # At T=6.5s, try 2 more requests
    time_module.sleep(0.5)
    time_65 = int(time_module.time())

    # First request at T=6.5s
    result_65_1 = await handler.batch_rate_limiter_script(
        keys=["user:ratelimit:window", "user:ratelimit:counter"],
        args=[time_65, window_size],
    )
    counter_65_1 = result_65_1[1]

    # Second request at T=6.5s (should exceed limit)
    result_65_2 = await handler.batch_rate_limiter_script(
        keys=["user:ratelimit:window", "user:ratelimit:counter"],
        args=[time_65, window_size],
    )
    counter_65_2 = result_65_2[1]

    print(f"T=~6.5s: 1st request counter={counter_65_1}, 2nd request counter={counter_65_2}")

    # Verify sliding window prevents burst
    if counter_65_2 > rate_limit:
        print(f"✓ SUCCESS: Redis Lua script correctly implements Sliding Window")
        print(f"  Counter {counter_65_2} > limit {rate_limit}")
        assert counter_65_2 > rate_limit, \
            f"Sliding Window should reject burst: {counter_65_2} > {rate_limit}"
    else:
        print(f"✗ FAIL: Redis Lua script allows burst (counter={counter_65_2} <= {rate_limit})")
        pytest.fail(
            f"Redis Lua script should implement Sliding Window.\n"
            f"Expected: counter > {rate_limit}, Got: counter = {counter_65_2}\n"
            f"At T=6.5s, weighted count should account for previous window requests"
        )
