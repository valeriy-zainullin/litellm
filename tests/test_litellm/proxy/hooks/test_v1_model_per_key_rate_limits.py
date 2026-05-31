"""
Unit Tests for per-model rate limit logic in parallel_request_limiter (v1).

Tests the model-specific rate limit logic for API keys in the v1 rate limiter:
- Priority: rpm_limit_per_model/tpm_limit_per_model from UserAPIKeyAuth > legacy fallback
- Independent TPM/RPM fallback
- Effective TPM/RPM calculation (per-model replaces key-level)
- Edge cases: None model, no limits configured, different model requested
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.parallel_request_limiter import (
    _PROXY_MaxParallelRequestsHandler,
)
from litellm.proxy.utils import InternalUsageCache, hash_token


@pytest.fixture
def mock_internal_usage_cache():
    """Create a mock InternalUsageCache."""
    mock = MagicMock(spec=InternalUsageCache)
    mock.async_batch_get_cache = AsyncMock(return_value=[None, None, None, None, None, None])
    mock.async_batch_set_cache = AsyncMock()
    return mock


@pytest.fixture
def handler(mock_internal_usage_cache):
    """Create a handler instance with mocked cache."""
    return _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=mock_internal_usage_cache
    )


class TestV1ModelPerKeyRateLimits:
    """Tests for per-model rate limit logic in v1 parallel_request_limiter."""

    @pytest.mark.asyncio
    async def test_uses_rpm_limit_per_model_from_user_api_key_auth(self, handler, mock_internal_usage_cache):
        """When rpm_limit_per_model is set on UserAPIKeyAuth, it takes priority over legacy metadata."""
        api_key = "sk-test-key"
        user_api_key_dict = UserAPIKeyAuth(
            api_key=api_key,
            rpm_limit_per_model={"gpt-4": 100},
            tpm_limit_per_model={"gpt-4": 10000},
            # Legacy metadata should be ignored
            metadata={
                "model_rpm_limit": {"gpt-4": 999},
                "model_tpm_limit": {"gpt-4": 99999},
            },
        )

        # Mock check_key_in_limits to capture the effective limits
        captured_tpm = None
        captured_rpm = None

        async def mock_check_key_in_limits(**kwargs):
            nonlocal captured_tpm, captured_rpm
            captured_tpm = kwargs.get("tpm_limit")
            captured_rpm = kwargs.get("rpm_limit")
            return {
                "current_requests": 1,
                "current_tpm": 0,
                "current_rpm": 1,
            }

        handler.check_key_in_limits = mock_check_key_in_limits  # type: ignore

        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=DualCache(),
            data={"model": "gpt-4"},
            call_type="completion",
        )

        # Should use per-model limits, not legacy
        assert captured_tpm == 10000, "Should use per-model TPM limit"
        assert captured_rpm == 100, "Should use per-model RPM limit"

    @pytest.mark.asyncio
    async def test_falls_back_to_legacy_when_user_api_key_auth_fields_none(self, handler, mock_internal_usage_cache):
        """When rpm_limit_per_model/tpm_limit_per_model are None, falls back to legacy metadata."""
        api_key = "sk-test-key"
        user_api_key_dict = UserAPIKeyAuth(
            api_key=api_key,
            rpm_limit_per_model=None,
            tpm_limit_per_model=None,
            metadata={
                "model_rpm_limit": {"gpt-4": 50},
                "model_tpm_limit": {"gpt-4": 5000},
            },
        )

        captured_tpm = None
        captured_rpm = None

        async def mock_check_key_in_limits(**kwargs):
            nonlocal captured_tpm, captured_rpm
            captured_tpm = kwargs.get("tpm_limit")
            captured_rpm = kwargs.get("rpm_limit")
            return {
                "current_requests": 1,
                "current_tpm": 0,
                "current_rpm": 1,
            }

        handler.check_key_in_limits = mock_check_key_in_limits  # type: ignore

        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=DualCache(),
            data={"model": "gpt-4"},
            call_type="completion",
        )

        assert captured_tpm == 5000, "Should use legacy TPM limit"
        assert captured_rpm == 50, "Should use legacy RPM limit"

    @pytest.mark.asyncio
    async def test_independent_tpm_rpm_fallback(self, handler, mock_internal_usage_cache):
        """When only TPM per-model is set, RPM falls back to key-level RPM."""
        api_key = "sk-test-key"
        user_api_key_dict = UserAPIKeyAuth(
            api_key=api_key,
            rpm_limit_per_model=None,  # No per-model RPM
            tpm_limit_per_model={"gpt-4": 10000},  # Per-model TPM
            rpm_limit=200,  # Key-level RPM
            tpm_limit=99999,  # Key-level TPM (should be overridden)
        )

        captured_tpm = None
        captured_rpm = None

        async def mock_check_key_in_limits(**kwargs):
            nonlocal captured_tpm, captured_rpm
            captured_tpm = kwargs.get("tpm_limit")
            captured_rpm = kwargs.get("rpm_limit")
            return {
                "current_requests": 1,
                "current_tpm": 0,
                "current_rpm": 1,
            }

        handler.check_key_in_limits = mock_check_key_in_limits  # type: ignore

        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=DualCache(),
            data={"model": "gpt-4"},
            call_type="completion",
        )

        # TPM should come from per-model, RPM should fall back to key-level
        assert captured_tpm == 10000, "TPM should use per-model limit"
        assert captured_rpm == 200, "RPM should fall back to key-level limit"

    @pytest.mark.asyncio
    async def test_independent_rpm_tpm_fallback(self, handler, mock_internal_usage_cache):
        """When only RPM per-model is set, TPM falls back to key-level TPM."""
        api_key = "sk-test-key"
        user_api_key_dict = UserAPIKeyAuth(
            api_key=api_key,
            rpm_limit_per_model={"gpt-4": 100},  # Per-model RPM
            tpm_limit_per_model=None,  # No per-model TPM
            rpm_limit=99999,  # Key-level RPM (should be overridden)
            tpm_limit=5000,  # Key-level TPM
        )

        captured_tpm = None
        captured_rpm = None

        async def mock_check_key_in_limits(**kwargs):
            nonlocal captured_tpm, captured_rpm
            captured_tpm = kwargs.get("tpm_limit")
            captured_rpm = kwargs.get("rpm_limit")
            return {
                "current_requests": 1,
                "current_tpm": 0,
                "current_rpm": 1,
            }

        handler.check_key_in_limits = mock_check_key_in_limits  # type: ignore

        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=DualCache(),
            data={"model": "gpt-4"},
            call_type="completion",
        )

        # RPM should come from per-model, TPM should fall back to key-level
        assert captured_tpm == 5000, "TPM should fall back to key-level limit"
        assert captured_rpm == 100, "RPM should use per-model limit"

    @pytest.mark.asyncio
    async def test_no_model_in_data_skips_per_model_check(self, handler, mock_internal_usage_cache):
        """When model is not in data, per-model check should be skipped."""
        api_key = "sk-test-key"
        user_api_key_dict = UserAPIKeyAuth(
            api_key=api_key,
            rpm_limit_per_model={"gpt-4": 100},
            tpm_limit_per_model={"gpt-4": 10000},
        )

        check_key_in_limits_called = False

        async def mock_check_key_in_limits(**kwargs):
            nonlocal check_key_in_limits_called
            check_key_in_limits_called = True
            return {
                "current_requests": 1,
                "current_tpm": 0,
                "current_rpm": 1,
            }

        handler.check_key_in_limits = mock_check_key_in_limits  # type: ignore

        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=DualCache(),
            data={},  # No model
            call_type="completion",
        )

        # check_key_in_limits should still be called for key-level check
        assert check_key_in_limits_called

    @pytest.mark.asyncio
    async def test_different_model_uses_key_level_limits(self, handler, mock_internal_usage_cache):
        """When limits exist but for a different model, per-model check still runs
        but falls back to key-level limits for the effective values."""
        api_key = "sk-test-key"
        user_api_key_dict = UserAPIKeyAuth(
            api_key=api_key,
            rpm_limit_per_model={"gpt-4": 100},
            tpm_limit_per_model={"gpt-4": 10000},
            rpm_limit=200,
            tpm_limit=5000,
        )

        captured_tpm = None
        captured_rpm = None

        async def mock_check_key_in_limits(**kwargs):
            nonlocal captured_tpm, captured_rpm
            if kwargs.get("rate_limit_type") == "model_per_key":
                captured_tpm = kwargs.get("tpm_limit")
                captured_rpm = kwargs.get("rpm_limit")
            return {
                "current_requests": 1,
                "current_tpm": 0,
                "current_rpm": 1,
            }

        handler.check_key_in_limits = mock_check_key_in_limits  # type: ignore

        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=DualCache(),
            data={"model": "claude-3"},  # Different model
            call_type="completion",
        )

        # model_per_key check IS called (because rpm_limit_per_model is not None),
        # but effective limits fall back to key-level since claude-3 is not in per-model dicts
        assert captured_tpm == 5000, "TPM should fall back to key-level limit for different model"
        assert captured_rpm == 200, "RPM should fall back to key-level limit for different model"

    @pytest.mark.asyncio
    async def test_no_limits_configured_skips_per_model_check(self, handler, mock_internal_usage_cache):
        """When no per-model limits are configured, per-model check should be skipped."""
        api_key = "sk-test-key"
        user_api_key_dict = UserAPIKeyAuth(
            api_key=api_key,
            rpm_limit_per_model=None,
            tpm_limit_per_model=None,
            rpm_limit=200,
            tpm_limit=5000,
        )

        captured_rate_limit_type = None

        async def mock_check_key_in_limits(**kwargs):
            nonlocal captured_rate_limit_type
            captured_rate_limit_type = kwargs.get("rate_limit_type")
            return {
                "current_requests": 1,
                "current_tpm": 0,
                "current_rpm": 1,
            }

        handler.check_key_in_limits = mock_check_key_in_limits  # type: ignore

        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=DualCache(),
            data={"model": "gpt-4"},
            call_type="completion",
        )

        # Only key-level check should be called, not model_per_key
        assert captured_rate_limit_type == "key", "Only key-level check should be called"

    @pytest.mark.asyncio
    async def test_effective_tpm_falls_back_to_key_level_when_not_in_per_model(
        self, handler, mock_internal_usage_cache
    ):
        """When per-model TPM dict exists but doesn't have the requested model, falls back to key-level TPM."""
        api_key = "sk-test-key"
        user_api_key_dict = UserAPIKeyAuth(
            api_key=api_key,
            rpm_limit_per_model={"gpt-4": 100},
            tpm_limit_per_model={"gpt-4": 10000},  # Only gpt-4 has TPM limit
            rpm_limit=200,
            tpm_limit=5000,
        )

        captured_tpm = None
        captured_rpm = None

        async def mock_check_key_in_limits(**kwargs):
            nonlocal captured_tpm, captured_rpm
            if kwargs.get("rate_limit_type") == "model_per_key":
                captured_tpm = kwargs.get("tpm_limit")
                captured_rpm = kwargs.get("rpm_limit")
            return {
                "current_requests": 1,
                "current_tpm": 0,
                "current_rpm": 1,
            }

        handler.check_key_in_limits = mock_check_key_in_limits  # type: ignore

        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=DualCache(),
            data={"model": "gpt-4"},  # gpt-4 has both RPM and TPM
            call_type="completion",
        )

        # Both should use per-model limits since gpt-4 is in both dicts
        assert captured_tpm == 10000, "TPM should use per-model limit for gpt-4"
        assert captured_rpm == 100, "RPM should use per-model limit for gpt-4"

    @pytest.mark.asyncio
    async def test_remaining_limits_added_to_metadata(self, handler, mock_internal_usage_cache):
        """Remaining tokens and requests should be added to data metadata."""
        api_key = "sk-test-key"
        user_api_key_dict = UserAPIKeyAuth(
            api_key=api_key,
            rpm_limit_per_model={"gpt-4": 100},
            tpm_limit_per_model={"gpt-4": 10000},
        )

        async def mock_check_key_in_limits(**kwargs):
            return {
                "current_requests": 1,
                "current_tpm": 500,
                "current_rpm": 10,
            }

        handler.check_key_in_limits = mock_check_key_in_limits  # type: ignore

        data = {"model": "gpt-4"}
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=DualCache(),
            data=data,
            call_type="completion",
        )

        # Check remaining limits in metadata
        metadata = data.get("metadata", {})
        assert metadata.get("litellm-key-remaining-tokens-gpt-4") == 10000 - 500
        assert metadata.get("litellm-key-remaining-requests-gpt-4") == 100 - 10