"""
Alias module for restricted_zone_rules.py to maintain backwards compatibility.
"""

from restricted_zone_rules import (
    RestrictedStatus,
    RestrictedZoneRuleModule,
    get_utc_now,
)

__all__ = ["RestrictedStatus", "RestrictedZoneRuleModule", "get_utc_now"]
