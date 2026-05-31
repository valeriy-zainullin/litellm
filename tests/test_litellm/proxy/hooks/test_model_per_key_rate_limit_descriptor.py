"""
Unit Tests for _add_model_per_key_rate_limit_descriptor in parallel_request_limiter_v3.

Tests the model-specific rate limit descriptor logic for API keys:
- Priority: rpm_limit_per_model/tpm_limit_per_model from UserAPIKeyAuth > legacy fallback
- Independent TPM/RPM fallback
- Key-level descriptor removal when per-model limits are set
- Edge cases: None model, no limits configured, different model requested
"""

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    _PROXY_MaxParallelRequestsHandler_v3 as _PROXY_MaxParallelRequestsHandler,
    RateLimitDescriptor,
)
from litellm.proxy.utils import InternalUsageCache, hash_token


@pytest.fixture
def handler():
    """Create a handler instance for testing."""
    return _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache())
    )


def _find_descriptor(
    descriptors: List[RateLimitDescriptor], key: str
) -> Optional[RateLimitDescriptor]:
    """Helper to find a descriptor by key."""
    for d in descriptors:
        if d["key"] == key:
            return d
    return None


class TestAddModelPerKeyRateLimitDescriptor:
    """Tests for _add_model_per_key_rate_limit_descriptor."""

    # ── Priority: UserAPIKeyAuth fields ──────────────────────────────

    def test_uses_rpm_limit_per_model_from_user_api_key_auth(self, handler):
        """When rpm_limit_per_model is set on UserAPIKeyAuth, it takes priority."""
        api_key = "sk-test-key"
        user_api_key_dict = UserAPIKeyAuth(
            api_key=api_key,
            rpm_limit_per_model={"gpt-4": 100},
            # Legacy metadata should be ignored when rpm_limit_per_model is set
            metadata={"model_rpm_limit": {"gpt-4": 999}},
        )
        descriptors: List[RateLimitDescriptor] = []

        handler._add_model_per_key_rate_limit_descriptor(
            user_api_key_dict=user_api_key_dict,
            requested_model="gpt-4",
            descriptors=descriptors,
        )

        model_per_key = _find_descriptor(descriptors, "model_per_key")
        assert model_per_key is not None
        assert model_per_key["rate_limit"]["requests_per_unit"] == 100

    def test_uses_tpm_limit_per_model_from_user_api_key_auth(self, handler):
        """When tpm_limit_per_model is set on UserAPIKeyAuth, it takes priority."""
        api_key = "sk-test-key"
        user_api_key_dict = UserAPIKeyAuth(
            api_key=api_key,
            tpm_limit_per_model={"gpt-4": 10000},
            metadata={"model_tpm_limit": {"gpt-4": 99999}},
        )
        descriptors: List[RateLimitDescriptor] = []

        handler._add_model_per_key_rate_limit_descriptor(
            user_api_key_dict=user_api_key_dict,
            requested_model="gpt-4",
            descriptors=descriptors,
        )

        model_per_key = _find_descriptor(descriptors, "model_per_key")
        assert model_per_key is not None
        assert model_per_key["rate_limit"]["tokens_per_unit"] == 10000

    # ── Fallback to legacy lookup ────────────────────────────────────

    def test_falls_back_to_legacy_rpm_when_rpm_limit_per_model_is_none(
        self, handler
    ):
        """When rpm_limit_per_model is None, falls back to get_key_model_rpm_limit."""
        api_key = "sk-test-key"
        user_api_key_dict = UserAPIKeyAuth(
            api_key=api_key,
            rpm_limit_per_model=None,
            tpm_limit_per_model={"gpt-4": 10000},
            metadata={"model_rpm_limit": {"gpt-4": 200}},
        )
        descriptors: List[RateLimitDescriptor] = []

        handler._add_model_per_key_rate_limit_descriptor(
            user_api_key_dict=user_api_key_dict,
            requested_model="gpt-4",
            descriptors=descriptors,
        )

        model_per_key = _find_descriptor(descriptors, "model_per_key")
        assert model_per_key is not None
        # RPM should come from legacy metadata fallback
        assert model_per_key["rate_limit"]["requests_per_unit"] == 200
        # TPM should come from tpm_limit_per_model
        assert model_per_key["rate_limit"]["tokens_per_unit"] == 10000

    def test_falls_back_to_legacy_tpm_when_tpm_limit_per_model_is_none(
        self, handler
    ):
        """When tpm_limit_per_model is None, falls back to get_key_model_tpm_limit."""
        api_key = "sk-test-key"
        user_api_key_dict = UserAPIKeyAuth(
            api_key=api_key,
            rpm_limit_per_model={"gpt-4": 100},
            tpm_limit_per_model=None,
            metadata={"model_tpm_limit": {"gpt-4": 5000}},
        )
        descriptors: List[RateLimitDescriptor] = []

        handler._add_model_per_key_rate_limit_descriptor(
            user_api_key_dict=user_api_key_dict,
            requested_model="gpt-4",
            descriptors=descriptors,
        )

        model_per_key = _find_descriptor(descriptors, "model_per_key")
        assert model_per_key is not None
        # RPM should come from rpm_limit_per_model
        assert model_per_key["rate_limit"]["requests_per_unit"] == 100
        # TPM should come from legacy metadata fallback
        assert model_per_key["rate_limit"]["tokens_per_unit"] == 5000

    # ── Independent TPM/RPM fallback ─────────────────────────────────

    def test_independent_tpm_rpm_fallback_rpm_only(self, handler):
        """When only RPM per-model is set, TPM falls back independently."""
        api_key = "sk-test-key"
        user_api_key_dict = UserAPIKeyAuth(
            api_key=api_key,
            rpm_limit_per_model={"gpt-4": 100},
            tpm_limit_per_model=None,
            # No legacy TPM either
        )
        descriptors: List[RateLimitDescriptor] = []

        handler._add_model_per_key_rate_limit_descriptor(
            user_api_key_dict=user_api_key_dict,
            requested_model="gpt-4",
            descriptors=descriptors,
        )

        model_per_key = _find_descriptor(descriptors, "model_per_key")
        assert model_per_key is not None
        assert model_per_key["rate_limit"]["requests_per_unit"] == 100
        assert model_per_key["rate_limit"]["tokens_per_unit"] is None

    def test_independent_tpm_rpm_fallback_tpm_only(self, handler):
        """When only TPM per-model is set, RPM falls back independently."""
        api_key = "sk-test-key"
        user_api_key_dict = UserAPIKeyAuth(
            api_key=api_key,
            rpm_limit_per_model=None,
            tpm_limit_per_model={"gpt-4": 10000},
        )
        descriptors: List[RateLimitDescriptor] = []

        handler._add_model_per_key_rate_limit_descriptor(
            user_api_key_dict=user_api_key_dict,
            requested_model="gpt-4",
            descriptors=descriptors,
        )

        model_per_key = _find_descriptor(descriptors, "model_per_key")
        assert model_per_key is not None
        assert model_per_key["rate_limit"]["requests_per_unit"] is None
        assert model_per_key["rate_limit"]["tokens_per_unit"] == 10000

    # ── Key-level descriptor removal ─────────────────────────────────

    def test_removes_key_level_descriptor_when_per_model_limits_set(self, handler):
        """When per-model limits are set, the key-level api_key descriptor is removed."""
        api_key = "sk-test-key"
        user_api_key_dict = UserAPIKeyAuth(
            api_key=api_key,
            rpm_limit_per_model={"gpt-4": 100},
            tpm_limit_per_model={"gpt-4": 10000},
        )
        # UserAPIKeyAuth hashes the api_key, so we need to use the hashed value
        hashed_key = user_api_key_dict.api_key
        descriptors: List[RateLimitDescriptor] = [
            {
                "key": "api_key",
                "value": hashed_key,
                "rate_limit": {
                    "requests_per_unit": 1000,
                    "tokens_per_unit": 100000,
                    "window_size": 2,
                },
            }
        ]

        handler._add_model_per_key_rate_limit_descriptor(
            user_api_key_dict=user_api_key_dict,
            requested_model="gpt-4",
            descriptors=descriptors,
        )

        # Key-level descriptor should be removed
        api_key_desc = _find_descriptor(descriptors, "api_key")
        assert api_key_desc is None, (
            "Key-level descriptor should be removed when per-model limits are set"
        )

        # Model-per-key descriptor should be present
        model_per_key = _find_descriptor(descriptors, "model_per_key")
        assert model_per_key is not None

    def test_does_not_remove_key_level_descriptor_for_different_key(self, handler):
        """Only the matching api_key descriptor is removed, not others."""
        api_key = "sk-test-key"
        other_key = "sk-other-key"
        user_api_key_dict = UserAPIKeyAuth(
            api_key=api_key,
            rpm_limit_per_model={"gpt-4": 100},
            tpm_limit_per_model={"gpt-4": 10000},
        )
        descriptors: List[RateLimitDescriptor] = [
            {
                "key": "api_key",
                "value": other_key,  # Different key (not hashed, so won't match)
                "rate_limit": {
                    "requests_per_unit": 1000,
                    "tokens_per_unit": 100000,
                    "window_size": 2,
                },
            }
        ]

        handler._add_model_per_key_rate_limit_descriptor(
            user_api_key_dict=user_api_key_dict,
            requested_model="gpt-4",
            descriptors=descriptors,
        )

        # Other key's descriptor should remain (it's a different value)
        api_key_desc = _find_descriptor(descriptors, "api_key")
        assert api_key_desc is not None
        assert api_key_desc["value"] == other_key

    def test_does_not_remove_key_level_descriptor_when_no_limits_for_model(
        self, handler
    ):
        """When per-model limits exist but not for the requested model, key-level descriptor stays."""
        api_key = "sk-test-key"
        user_api_key_dict = UserAPIKeyAuth(
            api_key=api_key,
            rpm_limit_per_model={"gpt-4": 100},
            tpm_limit_per_model={"gpt-4": 10000},
        )
        hashed_key = user_api_key_dict.api_key
        descriptors: List[RateLimitDescriptor] = [
            {
                "key": "api_key",
                "value": hashed_key,
                "rate_limit": {
                    "requests_per_unit": 1000,
                    "tokens_per_unit": 100000,
                    "window_size": 2,
                },
            }
        ]

        handler._add_model_per_key_rate_limit_descriptor(
            user_api_key_dict=user_api_key_dict,
            requested_model="gpt-3.5-turbo",  # Different model
            descriptors=descriptors,
        )

        # Key-level descriptor should remain
        api_key_desc = _find_descriptor(descriptors, "api_key")
        assert api_key_desc is not None

    # ── Edge cases ───────────────────────────────────────────────────

    def test_no_model_returns_early(self, handler):
        """When requested_model is None, the method returns early."""
        api_key = "sk-test-key"
        user_api_key_dict = UserAPIKeyAuth(
            api_key=api_key,
            rpm_limit_per_model={"gpt-4": 100},
        )
        descriptors: List[RateLimitDescriptor] = []

        handler._add_model_per_key_rate_limit_descriptor(
            user_api_key_dict=user_api_key_dict,
            requested_model=None,
            descriptors=descriptors,
        )

        assert len(descriptors) == 0

    def test_no_limits_configured_returns_early(self, handler):
        """When no limits are configured at all, the method returns early."""
        api_key = "sk-test-key"
        user_api_key_dict = UserAPIKeyAuth(
            api_key=api_key,
            rpm_limit_per_model=None,
            tpm_limit_per_model=None,
        )
        descriptors: List[RateLimitDescriptor] = []

        handler._add_model_per_key_rate_limit_descriptor(
            user_api_key_dict=user_api_key_dict,
            requested_model="gpt-4",
            descriptors=descriptors,
        )

        assert len(descriptors) == 0

    def test_different_model_no_match_returns_early(self, handler):
        """When limits exist but for a different model, returns early."""
        api_key = "sk-test-key"
        user_api_key_dict = UserAPIKeyAuth(
            api_key=api_key,
            rpm_limit_per_model={"gpt-4": 100},
            tpm_limit_per_model={"gpt-4": 10000},
        )
        descriptors: List[RateLimitDescriptor] = []

        handler._add_model_per_key_rate_limit_descriptor(
            user_api_key_dict=user_api_key_dict,
            requested_model="claude-3",  # Not in limits
            descriptors=descriptors,
        )

        assert len(descriptors) == 0

    def test_descriptor_value_combines_key_and_model(self, handler):
        """The descriptor value should be '{hashed_api_key}:{model}'."""
        api_key = "sk-test-key"
        user_api_key_dict = UserAPIKeyAuth(
            api_key=api_key,
            rpm_limit_per_model={"gpt-4": 100},
            tpm_limit_per_model={"gpt-4": 10000},
        )
        # UserAPIKeyAuth hashes the api_key
        hashed_key = user_api_key_dict.api_key
        descriptors: List[RateLimitDescriptor] = []

        handler._add_model_per_key_rate_limit_descriptor(
            user_api_key_dict=user_api_key_dict,
            requested_model="gpt-4",
            descriptors=descriptors,
        )

        model_per_key = _find_descriptor(descriptors, "model_per_key")
        assert model_per_key is not None
        assert model_per_key["value"] == f"{hashed_key}:gpt-4"

    def test_descriptor_has_window_size(self, handler):
        """The descriptor should include window_size from the handler."""
        api_key = "sk-test-key"
        user_api_key_dict = UserAPIKeyAuth(
            api_key=api_key,
            rpm_limit_per_model={"gpt-4": 100},
        )
        descriptors: List[RateLimitDescriptor] = []

        handler._add_model_per_key_rate_limit_descriptor(
            user_api_key_dict=user_api_key_dict,
            requested_model="gpt-4",
            descriptors=descriptors,
        )

        model_per_key = _find_descriptor(descriptors, "model_per_key")
        assert model_per_key is not None
        assert model_per_key["rate_limit"]["window_size"] == handler.window_size

    def test_empty_dict_limits_returns_early(self, handler):
        """When limits are empty dicts (not None), should still check correctly."""
        api_key = "sk-test-key"
        user_api_key_dict = UserAPIKeyAuth(
            api_key=api_key,
            rpm_limit_per_model={},  # Empty dict
            tpm_limit_per_model={},  # Empty dict
        )
        descriptors: List[RateLimitDescriptor] = []

        handler._add_model_per_key_rate_limit_descriptor(
            user_api_key_dict=user_api_key_dict,
            requested_model="gpt-4",
            descriptors=descriptors,
        )

        # Empty dicts mean no model matches, so no descriptor added
        assert len(descriptors) == 0

    def test_legacy_fallback_uses_metadata(self, handler):
        """When both UserAPIKeyAuth fields are None, falls back to metadata."""
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
        descriptors: List[RateLimitDescriptor] = []

        handler._add_model_per_key_rate_limit_descriptor(
            user_api_key_dict=user_api_key_dict,
            requested_model="gpt-4",
            descriptors=descriptors,
        )

        model_per_key = _find_descriptor(descriptors, "model_per_key")
        assert model_per_key is not None
        assert model_per_key["rate_limit"]["requests_per_unit"] == 50
        assert model_per_key["rate_limit"]["tokens_per_unit"] == 5000

    def test_legacy_fallback_team_metadata(self, handler):
        """Falls back to team_metadata when key metadata has no model limits."""
        api_key = "sk-test-key"
        user_api_key_dict = UserAPIKeyAuth(
            api_key=api_key,
            rpm_limit_per_model=None,
            tpm_limit_per_model=None,
            metadata={"some_other_key": "value"},
            team_metadata={
                "model_rpm_limit": {"gpt-4": 75},
                "model_tpm_limit": {"gpt-4": 7500},
            },
        )
        descriptors: List[RateLimitDescriptor] = []

        handler._add_model_per_key_rate_limit_descriptor(
            user_api_key_dict=user_api_key_dict,
            requested_model="gpt-4",
            descriptors=descriptors,
        )

        model_per_key = _find_descriptor(descriptors, "model_per_key")
        assert model_per_key is not None
        assert model_per_key["rate_limit"]["requests_per_unit"] == 75
        assert model_per_key["rate_limit"]["tokens_per_unit"] == 7500

    def test_preserves_other_descriptors(self, handler):
        """Other descriptors (team, user, etc.) should be preserved."""
        api_key = "sk-test-key"
        user_api_key_dict = UserAPIKeyAuth(
            api_key=api_key,
            rpm_limit_per_model={"gpt-4": 100},
            tpm_limit_per_model={"gpt-4": 10000},
        )
        hashed_key = user_api_key_dict.api_key
        descriptors: List[RateLimitDescriptor] = [
            {
                "key": "team",
                "value": "team-123",
                "rate_limit": {
                    "requests_per_unit": 500,
                    "tokens_per_unit": 50000,
                    "window_size": 2,
                },
            },
            {
                "key": "api_key",
                "value": hashed_key,
                "rate_limit": {
                    "requests_per_unit": 1000,
                    "tokens_per_unit": 100000,
                    "window_size": 2,
                },
            },
        ]

        handler._add_model_per_key_rate_limit_descriptor(
            user_api_key_dict=user_api_key_dict,
            requested_model="gpt-4",
            descriptors=descriptors,
        )

        # Team descriptor should be preserved
        team_desc = _find_descriptor(descriptors, "team")
        assert team_desc is not None
        assert team_desc["value"] == "team-123"

        # Key-level descriptor should be removed
        api_key_desc = _find_descriptor(descriptors, "api_key")
        assert api_key_desc is None

        # Model-per-key descriptor should be present
        model_per_key = _find_descriptor(descriptors, "model_per_key")
        assert model_per_key is not None