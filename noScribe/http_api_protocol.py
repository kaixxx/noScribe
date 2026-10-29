"""Version of the HTTP contract shared by the desktop client and server.

This is deliberately independent of the backend plugin protocol.  Either
contract may evolve without forcing an unrelated version bump in the other.
"""

# Version 2 requires clients to discard an attempt's segments when the same
# task starts again (retry: true). Version 1 clients cannot safely consume that.
NOSCRIBE_HTTP_API_PROTOCOL_VERSION = 2
