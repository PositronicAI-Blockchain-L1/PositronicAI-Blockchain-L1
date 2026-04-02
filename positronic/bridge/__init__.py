"""Positronic - Cross-Chain Bridge Interface (Abstraction Layer)

Provides data structures, state machines, and in-memory simulation for
cross-chain operations (hash anchoring and lock/mint token bridging).

NOTE: No external blockchain connectivity is implemented yet.  All
operations run against in-memory data structures.  The interfaces are
designed so that real chain connectors (web3.py, Bitcoin RPC, etc.)
can be plugged in without changing the public API.
"""
