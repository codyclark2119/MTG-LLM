"""Password gate for the chat surface, credentials supplied by environment.

A separate module from `chat_server.py` on purpose. `chat_server.py` holds a
4.4 GB model and imports mlx, so it is Apple-Silicon-only and can never run on
fly.io (plan finding #6: `requirements.txt` pins `mlx-metal`). The public
relay that eventually fronts it CAN run there, and it needs exactly this
logic — so the gate lives where both can import it and neither has to
reimplement it. A second copy of an auth check is how one copy ends up
missing a guard, which this repo has already paid for five times with
ordinary helpers.

Two environment variables, both intended to be fly secrets:

    CHAT_PASSWORD    the shared password. Required for any non-loopback bind.
    CHAT_SECRET_KEY  HMAC key signing the session cookie.

Set them with `fly secrets set` rather than in `fly.toml` — `fly.toml` is
committed and `[env]` values are visible in the app's public config, while
secrets are encrypted and injected at runtime.

Design notes that are security-relevant rather than stylistic:

  * the password is compared with `secrets.compare_digest`, so a wrong guess
    takes the same time as a right one and the comparison leaks no prefix;
  * the session cookie carries no identity to forge — it is `expiry.HMAC`,
    and the HMAC is over the expiry with the server's key, so a client cannot
    mint one or extend its own;
  * `CHAT_SECRET_KEY` absent means sessions are signed with a random key
    generated at startup, which is safe but logs everyone out on restart. A
    warning says so rather than letting it look like a bug later;
  * a missing `CHAT_PASSWORD` **fails closed** on any non-loopback bind. An
    auth layer whose default is "off" is the shape of every accidental public
    exposure, and this repo already has a 5.0 GB near-miss on that theme.
"""

import hashlib
import hmac
import os
import secrets
import threading
import time

COOKIE_NAME = "chat_session"
SESSION_TTL_S = 7 * 24 * 3600
_MAX_ATTEMPTS = 8
_LOCKOUT_S = 300

# The attempt table is bounded. Every entry is created by an UNAUTHENTICATED
# request, so its key space is whatever an attacker can put in front of the
# throttle -- and before this bound existed that was a memory leak reachable
# by anyone who could reach /api/login. 2048 is far above the number of real
# clients a personal service has and far below anything that matters for RSS.
_MAX_TRACKED_IPS = 2048

# Peers whose forwarded headers may be believed in `trust_proxy` mode. The
# deployment this exists for is `cloudflared` running ON THE SAME HOST and
# connecting to the loopback bind, so the trusted peer is loopback and nothing
# else. Widening this set is a decision about who may claim to be someone
# else, not a configuration detail.
TRUSTED_PEERS = frozenset({"127.0.0.1", "::1", "localhost"})

# Checked in this order, and only from a trusted peer. `CF-Connecting-IP` is a
# single value that Cloudflare SETS (overwriting whatever the client sent), so
# it cannot carry a forged list; X-Forwarded-For is a list and is read from the
# RIGHT, because each hop APPENDS the peer it saw -- a client that sends
# `X-Forwarded-For: 9.9.9.9` arrives at the origin as `9.9.9.9, <real client>`
# and the leftmost entry is precisely the attacker-controlled one. Reading the
# left is the bug this replaces.
_FORWARDED_HEADERS = ("cf-connecting-ip", "x-forwarded-for")


def client_ip(peer: str | None, headers=None, *, trust_proxy: bool = False,
              trusted_peers: frozenset[str] = TRUSTED_PEERS) -> str:
    """The identity rate limiting buckets on. NEVER used to authorize.

    `headers` is any case-insensitive mapping with `.get` (Starlette's
    `Headers`); a plain dict with lowercase keys works in tests.

    The rule is that a forwarded header is believed only when BOTH an operator
    has turned `trust_proxy` on AND the request actually arrived from a trusted
    peer. The previous version had neither condition: it read the first
    `X-Forwarded-For` value whenever one was present, which meant any client
    that could reach the port -- a LAN client, anyone on the machine -- could
    mint a fresh throttling identity per request by changing a header, and so
    could never be locked out at all. A header nothing strips is a client input.

    `trust_proxy` on with a non-trusted peer is not an error and not a fallback
    to the header: it returns the real peer. That is the case where someone
    reaches the LAN bind directly while a tunnel is also configured, and it is
    exactly the case the forged header wants.
    """
    peer = peer or "-"
    if not trust_proxy or peer not in trusted_peers:
        return peer
    if headers is not None:
        for name in _FORWARDED_HEADERS:
            raw = headers.get(name) or ""
            # Rightmost: the entry the trusted hop appended, not the one the
            # client chose. See _FORWARDED_HEADERS above.
            value = raw.split(",")[-1].strip()
            if value:
                return value
    return peer


class Auth:
    """Password check and signed-session issuing.

    `password=None` disables the gate entirely. Callers must only do that for
    a loopback bind; `require_password_for_public` enforces it so the decision
    is not left to each caller's discretion.
    """

    def __init__(self, password: str | None, secret_key: str | None = None,
                 ttl_s: int = SESSION_TTL_S):
        self.password = password
        self.ttl_s = ttl_s
        self.ephemeral_key = secret_key is None
        self.secret_key = (secret_key or secrets.token_hex(32)).encode()
        # ip -> (failed_count, first_failure_ts). Bounded: see _record_failure.
        self._attempts: dict[str, tuple[int, float]] = {}
        # `/api/login` is an `async def` and so runs on the event loop, but
        # `check_password` is a plain function a relay or a sync route may
        # reach from a worker thread. A dict is not safe to read-modify-write
        # across threads, and the failure would be a lost lockout increment --
        # invisible, and in the direction that favours the attacker.
        self._attempts_lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.password is not None

    def _sign(self, expiry: int) -> str:
        mac = hmac.new(self.secret_key, str(expiry).encode(), hashlib.sha256)
        return mac.hexdigest()

    def issue(self) -> str:
        expiry = int(time.time()) + self.ttl_s
        return f"{expiry}.{self._sign(expiry)}"

    def valid_session(self, cookie: str | None) -> bool:
        if not self.enabled:
            return True
        if not cookie or "." not in cookie:
            return False
        raw_expiry, _, sig = cookie.partition(".")
        try:
            expiry = int(raw_expiry)
        except ValueError:
            return False
        if expiry < time.time():
            return False
        # compare_digest, not ==, for the same reason the password is:
        # a byte-at-a-time comparison on a signature is forgeable given
        # enough attempts.
        return hmac.compare_digest(sig, self._sign(expiry))

    def _expire(self, now: float) -> None:
        """Drop every window that has run out, not just the one being asked about.

        The previous version expired an entry only when the SAME address came
        back, so an address that failed twice and never returned stayed forever
        and the table only ever grew. Expiry that triggers on a key's own
        traffic cannot collect the keys that stopped sending traffic, which is
        every key an identity-rotating attacker leaves behind.

        Caller holds `_attempts_lock`.
        """
        for ip in [k for k, (_, first) in self._attempts.items()
                   if now - first >= _LOCKOUT_S]:
            self._attempts.pop(ip, None)

    def _record_failure(self, ip: str, now: float) -> None:
        """Count one failure against `ip`, keeping the table under its bound.

        Eviction prefers addresses that are NOT currently locked out, oldest
        first. That ordering is the whole point: evicting indiscriminately
        would let an attacker who can mint identities flush a real lockout out
        of the table by filling it, turning the bound into a way to DEFEAT the
        throttle. Locked-out entries are the ones worth keeping, so they are
        evicted last and only when there is nothing else to drop.

        Caller holds `_attempts_lock`.
        """
        self._expire(now)
        count, first = self._attempts.get(ip, (0, now))
        self._attempts[ip] = (count + 1, first)

        if len(self._attempts) <= _MAX_TRACKED_IPS:
            return
        # `ip` itself is excluded from the candidates rather than skipped
        # inside the loop: skipping would evict one fewer than needed and
        # leave the table one over its bound on every such call.
        victims = sorted(
            ((k, v) for k, v in self._attempts.items() if k != ip),
            key=lambda kv: (kv[1][0] >= _MAX_ATTEMPTS, kv[1][1]))
        for victim, _ in victims[:len(self._attempts) - _MAX_TRACKED_IPS]:
            self._attempts.pop(victim, None)

    def locked_out(self, ip: str) -> bool:
        now = time.time()
        with self._attempts_lock:
            self._expire(now)
            count, first = self._attempts.get(ip, (0, 0.0))
            return count >= _MAX_ATTEMPTS and now - first < _LOCKOUT_S

    def tracked_identities(self) -> int:
        """Size of the attempt table. Exists so a test can assert the bound."""
        with self._attempts_lock:
            return len(self._attempts)

    def check_password(self, supplied: str, ip: str = "-") -> bool:
        """Constant-time password check with per-IP throttling.

        Throttling is deliberately crude: a shared password on a small
        personal service needs a brute-force speed bump, not an account
        system. It is NOT a substitute for a strong password, and the
        docstring says so because a lockout can otherwise read as one.

        `ip` must come from `client_ip`, which decides when a forwarded header
        may be believed. Passing a raw `X-Forwarded-For` value here is the bug
        that made the throttle unreachable.
        """
        if not self.enabled:
            return True
        if self.locked_out(ip):
            return False
        ok = hmac.compare_digest(supplied or "", self.password)
        with self._attempts_lock:
            if ok:
                self._attempts.pop(ip, None)
            else:
                self._record_failure(ip, time.time())
        return ok


def from_env(ttl_s: int = SESSION_TTL_S) -> Auth:
    return Auth(os.environ.get("CHAT_PASSWORD") or None,
                os.environ.get("CHAT_SECRET_KEY") or None, ttl_s=ttl_s)


def require_password_for_public(auth: Auth, host: str) -> list[str]:
    """Refuse to serve a non-loopback bind with no password.

    Returned as messages so the caller can print them and exit; raising from a
    helper that a future relay also imports would decide the exit behaviour
    for both. The check is on the BIND ADDRESS, because that is what actually
    determines reachability — not on a flag a caller might forget to pass.
    """
    problems = []
    if not auth.enabled and host not in ("127.0.0.1", "localhost", "::1"):
        problems.append(
            f"refusing to bind {host} with no CHAT_PASSWORD set.\n"
            "  Set one first:   export CHAT_PASSWORD='...'\n"
            "  On fly.io:       fly secrets set CHAT_PASSWORD='...' "
            "CHAT_SECRET_KEY=\"$(openssl rand -hex 32)\"\n"
            "  Loopback (127.0.0.1) is allowed without one.")
    return problems


def cookie_kwargs(secure: bool) -> dict:
    """Cookie flags. `secure` should be True whenever served over HTTPS.

    HttpOnly keeps the session out of `document.cookie`, so an injected script
    cannot read it; SameSite=Lax stops a third-party page from silently
    posting as the logged-in user.
    """
    return {"httponly": True, "samesite": "lax", "secure": secure,
            "max_age": SESSION_TTL_S, "path": "/"}
