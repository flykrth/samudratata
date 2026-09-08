"""
config_setup.py
───────────────
Sentinel-2 Level-2A — Deep Learning Project (2019-2024)

PURPOSE
  Securely load API credentials from a .env file and verify that
  authentication works against the selected backend:
    • "cdse"  – Copernicus Data Space Ecosystem (OData / OpenSearch)
    • "gee"   – Google Earth Engine Python API

USAGE
  1.  cp .env.example .env      # then fill in your real credentials
  2.  python config_setup.py    # authenticate & verify

The script NEVER downloads any imagery — it only confirms the
credentials are valid so the rest of the pipeline can rely on them.
"""

from __future__ import annotations

import os
import sys
import json
import textwrap
import logging
from pathlib import Path
from datetime import datetime

# ── Third-party (installed in .venv) ──────────────────────────────
try:
    from dotenv import load_dotenv, find_dotenv
except ImportError:
    sys.exit(
        "[FATAL] python-dotenv is not installed.\n"
        "        Run: pip install python-dotenv"
    )

# ─────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# 1.  Load .env
# ─────────────────────────────────────────────────────────────────
_dotenv_path = find_dotenv(usecwd=True)
if not _dotenv_path:
    log.warning(
        ".env file not found.  Copy .env.example → .env and fill in "
        "your credentials, then re-run this script."
    )
else:
    load_dotenv(_dotenv_path, override=False)
    log.info("Loaded environment variables from: %s", _dotenv_path)


# ─────────────────────────────────────────────────────────────────
# 2.  Config dataclass (pure stdlib)
# ─────────────────────────────────────────────────────────────────
class Config:
    """Reads and validates all project-level settings from os.environ."""

    # ── Auth backend ───────────────────────────────────────────────
    AUTH_BACKEND: str = os.getenv("AUTH_BACKEND", "gee").lower()

    # ── CDSE credentials ──────────────────────────────────────────
    CDSE_USER: str = os.getenv("CDSE_USER", "")
    CDSE_PASSWORD: str = os.getenv("CDSE_PASSWORD", "")

    # ── GEE credentials ───────────────────────────────────────────
    GEE_SERVICE_ACCOUNT: str = os.getenv("GEE_SERVICE_ACCOUNT", "")
    GEE_KEY_FILE: str = os.getenv("GEE_KEY_FILE", "")
    GEE_PROJECT: str = os.getenv("GEE_PROJECT", "")

    # ── Project settings ──────────────────────────────────────────
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", "./data/sentinel2"))
    DATE_START: str = os.getenv("DATE_START", "2019-01-01")
    DATE_END: str = os.getenv("DATE_END", "2024-12-31")
    MAX_CLOUD_COVER: int = int(os.getenv("MAX_CLOUD_COVER", "20"))

    # ── Validation ────────────────────────────────────────────────
    @classmethod
    def validate(cls) -> list[str]:
        """Return a list of human-readable validation errors (empty = OK)."""
        errors: list[str] = []

        if cls.AUTH_BACKEND not in {"cdse", "gee"}:
            errors.append(
                f"AUTH_BACKEND must be 'cdse' or 'gee', got '{cls.AUTH_BACKEND}'"
            )

        if cls.AUTH_BACKEND == "cdse":
            if not cls.CDSE_USER:
                errors.append("CDSE_USER is not set")
            if not cls.CDSE_PASSWORD:
                errors.append("CDSE_PASSWORD is not set")

        if cls.AUTH_BACKEND == "gee":
            if not cls.GEE_PROJECT:
                errors.append("GEE_PROJECT is not set")
            if cls.GEE_SERVICE_ACCOUNT:
                key_path = Path(cls.GEE_KEY_FILE)
                if not cls.GEE_KEY_FILE:
                    errors.append(
                        "GEE_SERVICE_ACCOUNT is set but GEE_KEY_FILE is missing"
                    )
                elif not key_path.exists():
                    errors.append(
                        f"GEE_KEY_FILE does not exist: {key_path}"
                    )
                else:
                    try:
                        with key_path.open() as fh:
                            key_data = json.load(fh)
                        if key_data.get("type") != "service_account":
                            errors.append(
                                "GEE_KEY_FILE does not look like a service-account "
                                "JSON key (missing or wrong 'type' field)"
                            )
                    except json.JSONDecodeError as exc:
                        errors.append(f"GEE_KEY_FILE is not valid JSON: {exc}")

        # Date range
        try:
            t0 = datetime.strptime(cls.DATE_START, "%Y-%m-%d")
            t1 = datetime.strptime(cls.DATE_END, "%Y-%m-%d")
            if t0 >= t1:
                errors.append("DATE_START must be before DATE_END")
        except ValueError as exc:
            errors.append(f"Invalid date format: {exc}")

        # Cloud cover
        if not 0 <= cls.MAX_CLOUD_COVER <= 100:
            errors.append("MAX_CLOUD_COVER must be between 0 and 100")

        return errors

    @classmethod
    def summary(cls) -> str:
        """One-line human-readable config dump (no secrets printed)."""
        return (
            f"backend={cls.AUTH_BACKEND}  "
            f"dates={cls.DATE_START}→{cls.DATE_END}  "
            f"cloud≤{cls.MAX_CLOUD_COVER}%  "
            f"output={cls.DATA_DIR}"
        )


# ─────────────────────────────────────────────────────────────────
# 3.  Authentication helpers
# ─────────────────────────────────────────────────────────────────

def authenticate_cdse() -> bool:
    """
    Verify CDSE credentials by requesting an OAuth2 token from the
    Copernicus IAM endpoint.
    Returns True on success, raises RuntimeError on failure.
    """
    try:
        import requests
    except ImportError:
        raise RuntimeError("requests is not installed: pip install requests")

    TOKEN_URL = (
        "https://identity.dataspace.copernicus.eu"
        "/auth/realms/CDSE/protocol/openid-connect/token"
    )
    log.info("CDSE → requesting OAuth2 token from %s", TOKEN_URL)

    resp = requests.post(
        TOKEN_URL,
        data={
            "client_id": "cdse-public",
            "grant_type": "password",
            "username": Config.CDSE_USER,
            "password": Config.CDSE_PASSWORD,
        },
        timeout=30,
    )

    if resp.status_code == 200:
        token_data = resp.json()
        expires_in = token_data.get("expires_in", "?")
        log.info(
            "CDSE ✓ Authentication successful  "
            "(token_type=%s, expires_in=%ss)",
            token_data.get("token_type", "?"),
            expires_in,
        )
        return True
    else:
        raise RuntimeError(
            f"CDSE authentication failed  "
            f"(HTTP {resp.status_code}): {resp.text[:300]}"
        )


def authenticate_gee() -> bool:
    """
    Authenticate with Google Earth Engine.

    Two flows are supported:
      • Service-account – when GEE_SERVICE_ACCOUNT + GEE_KEY_FILE are set
      • Interactive/ADC  – when only GEE_PROJECT is set
                           (prompts once, caches in ~/.config/earthengine/)

    Returns True on success, raises RuntimeError on failure.
    """
    try:
        import ee
    except ImportError:
        raise RuntimeError(
            "earthengine-api is not installed: pip install earthengine-api"
        )

    project = Config.GEE_PROJECT

    if Config.GEE_SERVICE_ACCOUNT and Config.GEE_KEY_FILE:
        # ── Service-account flow ──────────────────────────────────
        key_path = Path(Config.GEE_KEY_FILE)
        log.info(
            "GEE → service-account flow  (account=%s, key=%s)",
            Config.GEE_SERVICE_ACCOUNT,
            key_path.name,
        )
        credentials = ee.ServiceAccountCredentials(
            Config.GEE_SERVICE_ACCOUNT,
            str(key_path),
        )
        ee.Initialize(credentials=credentials, project=project)
    else:
        # ── Interactive / Application Default Credentials ─────────
        log.info(
            "GEE → interactive / ADC flow  (project=%s)  "
            "If not yet authenticated, run:  earthengine authenticate",
            project,
        )
        ee.Initialize(project=project)

    # ── Smoke-test: fetch a known S2 image metadata ───────────────
    test_img = ee.Image(
        "COPERNICUS/S2_SR_HARMONIZED"
        "/20190101T103351_20190101T103351_T32UMD"
    )
    info = test_img.select("B4").getInfo()

    if info and info.get("type") == "Image":
        log.info(
            "GEE ✓ Authentication successful  "
            "(verified access to COPERNICUS/S2_SR_HARMONIZED collection)"
        )
        return True
    else:
        raise RuntimeError(
            "GEE auth appeared to succeed but the test image "
            "could not be fetched.  Check project quota and permissions."
        )


# ─────────────────────────────────────────────────────────────────
# 4.  Main
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "═" * 62)
    print("  Sentinel-2 DL Project — Config & Auth Verification")
    print("═" * 62 + "\n")

    # ── Validate config ───────────────────────────────────────────
    errors = Config.validate()
    if errors:
        log.error("Configuration errors found:\n%s",
                  "\n".join(f"  • {e}" for e in errors))
        print(
            "\n[HINT] Copy .env.example to .env and fill in the "
            "required values, then re-run this script.\n"
        )
        sys.exit(1)

    log.info("Config OK  →  %s", Config.summary())

    # ── Create output directory ───────────────────────────────────
    Config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Data directory ready: %s", Config.DATA_DIR.resolve())

    # ── Authenticate ──────────────────────────────────────────────
    try:
        if Config.AUTH_BACKEND == "cdse":
            authenticate_cdse()
        else:
            authenticate_gee()
    except RuntimeError as exc:
        log.error("Authentication failed: %s", exc)
        sys.exit(1)
    except Exception as exc:
        log.exception("Unexpected error during authentication: %s", exc)
        sys.exit(1)

    # ── Summary banner ────────────────────────────────────────────
    print(
        textwrap.dedent(f"""
        ┌─────────────────────────────────────────────────────────┐
        │  ✓ Authentication verified — ready to download imagery  │
        │                                                         │
        │  Backend    : {Config.AUTH_BACKEND.upper():<40} │
        │  Date range : {Config.DATE_START} → {Config.DATE_END:<22} │
        │  Max clouds : {str(Config.MAX_CLOUD_COVER) + "%":<40} │
        │  Data dir   : {str(Config.DATA_DIR):<40} │
        └─────────────────────────────────────────────────────────┘
        """)
    )
    log.info(
        "config_setup.py completed.  "
        "Next step: run download_s2.py (not yet implemented)."
    )


if __name__ == "__main__":
    main()
