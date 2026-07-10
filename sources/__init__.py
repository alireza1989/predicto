"""Data-source plugin registry.

Each source implements the DataSource contract in sources/base.py. The Data
Scout agent enumerates REGISTRY to health-check, fetch, and score sources.
"""
from sources.espn_injuries import ESPNInjuriesSource

REGISTRY = {
    "espn_injuries": ESPNInjuriesSource(),
}
