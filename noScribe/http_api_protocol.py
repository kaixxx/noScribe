"""Version of the HTTP contract shared by the desktop client and server.

This is deliberately independent of the backend plugin protocol.  Either
contract may evolve without forcing an unrelated version bump in the other.
"""

NOSCRIBE_HTTP_API_PROTOCOL_VERSION = 1
