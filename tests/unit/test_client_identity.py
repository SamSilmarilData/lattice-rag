"""Client identity used for rate limiting and admin-key throttling.

The socket peer is the identity unless TRUSTED_PROXY_HOPS says how many reverse
proxies sit in front of the app; only then is X-Forwarded-For consulted.
"""

from __future__ import annotations

import pytest

from lattice_rag.api.client_identity import resolve_client_address


def test_no_trusted_proxies_ignores_forged_forwarded_for():
    assert resolve_client_address("203.0.113.9", "1.2.3.4", trusted_proxy_hops=0) == "203.0.113.9"


def test_no_trusted_proxies_without_peer_is_unknown():
    assert resolve_client_address(None, "1.2.3.4", trusted_proxy_hops=0) == "unknown"


def test_one_trusted_hop_takes_rightmost_entry_and_ignores_spoofed_left_entries():
    header = "6.6.6.6, 7.7.7.7, 198.51.100.4"
    assert resolve_client_address("10.0.0.1", header, trusted_proxy_hops=1) == "198.51.100.4"


def test_two_trusted_hops_take_second_entry_from_the_right():
    header = "6.6.6.6, 198.51.100.4, 10.0.0.2"
    assert resolve_client_address("10.0.0.1", header, trusted_proxy_hops=2) == "198.51.100.4"


@pytest.mark.parametrize("header", [None, "", "198.51.100.4"])
def test_header_shorter_than_trusted_hops_falls_back_to_peer(header):
    assert resolve_client_address("10.0.0.1", header, trusted_proxy_hops=2) == "10.0.0.1"


def test_blank_entries_are_not_counted_as_hops():
    assert resolve_client_address("10.0.0.1", " , 198.51.100.4 ,", trusted_proxy_hops=1) == (
        "198.51.100.4"
    )
