import base64
import hashlib

from cryptography.fernet import Fernet, InvalidToken, MultiFernet
from django.conf import settings
from django.db import models

ENCRYPTED_PREFIX = "enc::"


def _cipher():
    configured = [item.strip() for item in getattr(settings, "FIELD_ENCRYPTION_KEYS", []) if item.strip()]
    if configured:
        keys = [item.encode("ascii") for item in configured]
    else:
        digest = hashlib.sha256(settings.SECRET_KEY.encode("utf-8")).digest()
        keys = [base64.urlsafe_b64encode(digest)]
    return MultiFernet([Fernet(key) for key in keys])


class EncryptedTextField(models.TextField):
    """Fernet-encrypted text with transparent reads and legacy plaintext support."""

    def _decrypt(self, value):
        if not isinstance(value, str) or not value.startswith(ENCRYPTED_PREFIX):
            return value
        try:
            return _cipher().decrypt(value[len(ENCRYPTED_PREFIX):].encode("ascii")).decode("utf-8")
        except (InvalidToken, UnicodeDecodeError, ValueError) as exc:
            raise ValueError("Encrypted database value could not be decrypted") from exc

    def from_db_value(self, value, expression, connection):
        return self._decrypt(value)

    def to_python(self, value):
        return self._decrypt(value)

    def get_prep_value(self, value):
        value = super().get_prep_value(value)
        if value in (None, "") or str(value).startswith(ENCRYPTED_PREFIX):
            return value
        token = _cipher().encrypt(str(value).encode("utf-8")).decode("ascii")
        return f"{ENCRYPTED_PREFIX}{token}"
