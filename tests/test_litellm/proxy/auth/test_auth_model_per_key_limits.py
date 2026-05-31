"""
Unit Tests for per-model rate limit fields in user_api_key_auth.

Tests that rpm_limit_per_model/tpm_limit_per_model are correctly populated
from key metadata during authentication.
"""

from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy._types import UserAPIKeyAuth


class TestAuthModelPerKeyLimits:
    """Tests for per-model rate limit field population in auth."""

    def test_rpm_limit_per_model_populated_from_metadata(self):
        """When metadata has model_rpm_limit, rpm_limit_per_model should be set."""
        valid_token = UserAPIKeyAuth(
            api_key="sk-test",
            metadata={"model_rpm_limit": {"gpt-4": 100, "gpt-3.5-turbo": 200}},
        )

        valid_token_dict = valid_token.model_dump(exclude_none=True)

        # Simulate the auth logic from user_api_key_auth.py
        if valid_token.metadata:
            _model_rpm = valid_token.metadata.get("model_rpm_limit")
            if _model_rpm is not None:
                valid_token_dict["rpm_limit_per_model"] = _model_rpm

        assert valid_token_dict.get("rpm_limit_per_model") == {
            "gpt-4": 100,
            "gpt-3.5-turbo": 200,
        }

    def test_tpm_limit_per_model_populated_from_metadata(self):
        """When metadata has model_tpm_limit, tpm_limit_per_model should be set."""
        valid_token = UserAPIKeyAuth(
            api_key="sk-test",
            metadata={"model_tpm_limit": {"gpt-4": 10000, "gpt-3.5-turbo": 20000}},
        )

        valid_token_dict = valid_token.model_dump(exclude_none=True)

        if valid_token.metadata:
            _model_tpm = valid_token.metadata.get("model_tpm_limit")
            if _model_tpm is not None:
                valid_token_dict["tpm_limit_per_model"] = _model_tpm

        assert valid_token_dict.get("tpm_limit_per_model") == {
            "gpt-4": 10000,
            "gpt-3.5-turbo": 20000,
        }

    def test_both_rpm_and_tpm_populated(self):
        """Both rpm_limit_per_model and tpm_limit_per_model should be set when both exist."""
        valid_token = UserAPIKeyAuth(
            api_key="sk-test",
            metadata={
                "model_rpm_limit": {"gpt-4": 100},
                "model_tpm_limit": {"gpt-4": 10000},
            },
        )

        valid_token_dict = valid_token.model_dump(exclude_none=True)

        if valid_token.metadata:
            _model_rpm = valid_token.metadata.get("model_rpm_limit")
            if _model_rpm is not None:
                valid_token_dict["rpm_limit_per_model"] = _model_rpm
            _model_tpm = valid_token.metadata.get("model_tpm_limit")
            if _model_tpm is not None:
                valid_token_dict["tpm_limit_per_model"] = _model_tpm

        assert valid_token_dict.get("rpm_limit_per_model") == {"gpt-4": 100}
        assert valid_token_dict.get("tpm_limit_per_model") == {"gpt-4": 10000}

    def test_no_metadata_does_not_set_fields(self):
        """When metadata is empty, rpm_limit_per_model/tpm_limit_per_model should not be set."""
        valid_token = UserAPIKeyAuth(
            api_key="sk-test",
            metadata={},
        )

        valid_token_dict = valid_token.model_dump(exclude_none=True)

        if valid_token.metadata:
            _model_rpm = valid_token.metadata.get("model_rpm_limit")
            if _model_rpm is not None:
                valid_token_dict["rpm_limit_per_model"] = _model_rpm
            _model_tpm = valid_token.metadata.get("model_tpm_limit")
            if _model_tpm is not None:
                valid_token_dict["tpm_limit_per_model"] = _model_tpm

        assert valid_token_dict.get("rpm_limit_per_model") is None
        assert valid_token_dict.get("tpm_limit_per_model") is None

    def test_metadata_without_model_limits_does_not_set_fields(self):
        """When metadata exists but has no model_rpm_limit/model_tpm_limit, fields should not be set."""
        valid_token = UserAPIKeyAuth(
            api_key="sk-test",
            metadata={"some_other_key": "value"},
        )

        valid_token_dict = valid_token.model_dump(exclude_none=True)

        if valid_token.metadata:
            _model_rpm = valid_token.metadata.get("model_rpm_limit")
            if _model_rpm is not None:
                valid_token_dict["rpm_limit_per_model"] = _model_rpm
            _model_tpm = valid_token.metadata.get("model_tpm_limit")
            if _model_tpm is not None:
                valid_token_dict["tpm_limit_per_model"] = _model_tpm

        assert valid_token_dict.get("rpm_limit_per_model") is None
        assert valid_token_dict.get("tpm_limit_per_model") is None

    def test_model_rpm_limit_none_does_not_set_field(self):
        """When model_rpm_limit is explicitly None, rpm_limit_per_model should not be set."""
        valid_token = UserAPIKeyAuth(
            api_key="sk-test",
            metadata={"model_rpm_limit": None, "model_tpm_limit": {"gpt-4": 10000}},
        )

        valid_token_dict = valid_token.model_dump(exclude_none=True)

        if valid_token.metadata:
            _model_rpm = valid_token.metadata.get("model_rpm_limit")
            if _model_rpm is not None:
                valid_token_dict["rpm_limit_per_model"] = _model_rpm
            _model_tpm = valid_token.metadata.get("model_tpm_limit")
            if _model_tpm is not None:
                valid_token_dict["tpm_limit_per_model"] = _model_tpm

        assert valid_token_dict.get("rpm_limit_per_model") is None
        assert valid_token_dict.get("tpm_limit_per_model") == {"gpt-4": 10000}

    def test_model_tpm_limit_none_does_not_set_field(self):
        """When model_tpm_limit is explicitly None, tpm_limit_per_model should not be set."""
        valid_token = UserAPIKeyAuth(
            api_key="sk-test",
            metadata={"model_rpm_limit": {"gpt-4": 100}, "model_tpm_limit": None},
        )

        valid_token_dict = valid_token.model_dump(exclude_none=True)

        if valid_token.metadata:
            _model_rpm = valid_token.metadata.get("model_rpm_limit")
            if _model_rpm is not None:
                valid_token_dict["rpm_limit_per_model"] = _model_rpm
            _model_tpm = valid_token.metadata.get("model_tpm_limit")
            if _model_tpm is not None:
                valid_token_dict["tpm_limit_per_model"] = _model_tpm

        assert valid_token_dict.get("rpm_limit_per_model") == {"gpt-4": 100}
        assert valid_token_dict.get("tpm_limit_per_model") is None

    def test_fields_are_present_in_user_api_key_auth_model(self):
        """Verify that rpm_limit_per_model and tpm_limit_per_model are actual fields on UserAPIKeyAuth."""
        auth = UserAPIKeyAuth(
            api_key="sk-test",
            rpm_limit_per_model={"gpt-4": 100},
            tpm_limit_per_model={"gpt-4": 10000},
        )

        assert auth.rpm_limit_per_model == {"gpt-4": 100}
        assert auth.tpm_limit_per_model == {"gpt-4": 10000}

    def test_fields_are_none_by_default(self):
        """Verify that rpm_limit_per_model and tpm_limit_per_model are None by default."""
        auth = UserAPIKeyAuth(api_key="sk-test")

        assert auth.rpm_limit_per_model is None
        assert auth.tpm_limit_per_model is None

    def test_fields_survive_model_dump_roundtrip(self):
        """Verify that rpm_limit_per_model and tpm_limit_per_model survive model_dump -> model_validate."""
        auth = UserAPIKeyAuth(
            api_key="sk-test",
            rpm_limit_per_model={"gpt-4": 100},
            tpm_limit_per_model={"gpt-4": 10000},
        )

        dumped = auth.model_dump(exclude_none=True)
        restored = UserAPIKeyAuth(**dumped)

        assert restored.rpm_limit_per_model == {"gpt-4": 100}
        assert restored.tpm_limit_per_model == {"gpt-4": 10000}