"""Clerk JWT verification dependency for FastAPI."""

from __future__ import annotations

import structlog
import httpx
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from app.core.config import settings

logger = structlog.get_logger(__name__)

security = HTTPBearer()

# ── JWKS Cache ────────────────────────────────────────────────
_jwks_cache: dict | None = None


async def _get_jwks() -> dict:
    """Fetch Clerk's JWKS (JSON Web Key Set) and cache it."""
    global _jwks_cache
    if _jwks_cache is not None:
        return _jwks_cache

    async with httpx.AsyncClient() as client:
        resp = await client.get(settings.CLERK_JWKS_URL, timeout=10.0)
        resp.raise_for_status()
        _jwks_cache = resp.json()
        logger.info("clerk_jwks_fetched", num_keys=len(_jwks_cache.get("keys", [])))
        return _jwks_cache


def _find_signing_key(jwks: dict, token: str) -> dict:
    """Find the RSA public key matching the token's kid header."""
    unverified_header = jwt.get_unverified_header(token)
    kid = unverified_header.get("kid")

    for key in jwks.get("keys", []):
        if key.get("kid") == kid:
            return key

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unable to find matching signing key.",
    )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """
    FastAPI dependency that verifies a Clerk JWT and returns the user's
    Clerk ID (the ``sub`` claim).

    Usage::

        @router.get("/protected")
        async def protected(user_id: str = Depends(get_current_user)):
            ...
    """
    token = credentials.credentials

    try:
        jwks = await _get_jwks()
        rsa_key = _find_signing_key(jwks, token)

        payload = jwt.decode(
            token,
            rsa_key,
            algorithms=["RS256"],
            issuer=settings.CLERK_ISSUER_URL,
            options={
                "verify_aud": False,  # Clerk doesn't always set aud
                "verify_exp": True,
                "verify_iss": True,
            },
        )

        clerk_user_id: str | None = payload.get("sub")
        if not clerk_user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token missing 'sub' claim.",
            )

        return clerk_user_id

    except JWTError as exc:
        logger.warning("jwt_verification_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication token: {exc}",
        ) from exc
    except httpx.HTTPError as exc:
        logger.error("jwks_fetch_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to verify token — authentication service unavailable.",
        ) from exc
    try:
        jwks = await _get_jwks()
        rsa_key = _find_signing_key(jwks, token)

        payload = jwt.decode(
            token,
            rsa_key,
            algorithms=["RS256"],
            issuer=settings.CLERK_ISSUER_URL,
            options={
                "verify_aud": False,  # Clerk doesn't always set aud
                "verify_exp": True,
                "verify_iss": True,
            },
        )

        clerk_user_id: str | None = payload.get("sub")
        if not clerk_user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token missing 'sub' claim.",
            )

        return clerk_user_id

    except JWTError as exc:
        logger.warning("jwt_verification_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication token: {exc}",
        ) from exc
    except httpx.HTTPError as exc:
        logger.error("jwks_fetch_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to verify token — authentication service unavailable.",
        ) from exc
