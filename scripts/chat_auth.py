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
import time

COOKIE_NAME = "chat_session"
SESSION_TTL_S = 7 * 24 * 3600
_MAX_ATTEMPTS = 8
_LOCKOUT_S = 300


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
        # ip -> (failed_count, first_failure_ts). Bounded by eviction on read.
        self._attempts: dict[str, tuple[int, float]] = {}

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

    def locked_out(self, ip: str) -> bool:
        count, first = self._attempts.get(ip, (0, 0.0))
        if count >= _MAX_ATTEMPTS and time.time() - first < _LOCKOUT_S:
            return True
        if count and time.time() - first >= _LOCKOUT_S:
            self._attempts.pop(ip, None)
        return False

    def check_password(self, supplied: str, ip: str = "-") -> bool:
        """Constant-time password check with per-IP throttling.

        Throttling is deliberately crude: a shared password on a small
        personal service needs a brute-force speed bump, not an account
        system. It is NOT a substitute for a strong password, and the
        docstring says so because a lockout can otherwise read as one.
        """
        if not self.enabled:
            return True
        if self.locked_out(ip):
            return False
        ok = hmac.compare_digest(supplied or "", self.password)
        if ok:
            self._attempts.pop(ip, None)
        else:
            count, first = self._attempts.get(ip, (0, time.time()))
            self._attempts[ip] = (count + 1, first)
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
