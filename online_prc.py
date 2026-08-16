"""Causal, prefix-consistent PRC construction for online generation.

Unlike :mod:`prc`, this module never samples a length-n matrix and never
permutes its coordinates.  At coordinate ``i`` it either samples a free clean
bit or introduces one parity check whose pivot is ``i`` and whose remaining
coordinates are all strictly smaller than ``i``.  Consequently the same key
defines compatible checks for every realized prefix length.

The HMAC expansion below makes experiment artifacts reproducible and prevents
one prefix length from consuming randomness needed by another.  Integer seeds
are convenient for experiments; production key management should supply
uniform secret key bytes instead.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import math
import struct
from typing import Iterable, Sequence

import numpy as np


SCHEME = "online_causal_prc_v1"
SCHEDULE_VERSION = "rational_round_half_even_v1"
SUPPORT_SAMPLER_VERSION = "hmac_sha256_rejection_v1"
GENERATION_SAMPLER_VERSION = "document_position_inverse_cdf_v1"
KEY_SCHEMA_VERSION = 1
DEFAULT_ROW_RATE_NUMERATOR = 99
DEFAULT_ROW_RATE_DENOMINATOR = 100


def _seed_key(seed: int | bytes | bytearray | str) -> bytes:
    if isinstance(seed, int):
        if seed < 0:
            raise ValueError("seed must be nonnegative")
        width = max(1, (seed.bit_length() + 7) // 8)
        return seed.to_bytes(width, "big")
    if isinstance(seed, str):
        return seed.encode("utf-8")
    if isinstance(seed, (bytes, bytearray)):
        if not seed:
            raise ValueError("seed bytes must be nonempty")
        return bytes(seed)
    raise TypeError(f"unsupported seed type {type(seed).__name__}")


def _expand(key: bytes, domain: bytes, *values: int) -> bytes:
    message = bytearray(domain)
    for value in values:
        if int(value) < 0:
            raise ValueError("PRF integer inputs must be nonnegative")
        message.extend(struct.pack(">Q", int(value)))
    return hmac.new(key, bytes(message), hashlib.sha256).digest()


def _round_rational_half_even(numerator: int, denominator: int) -> int:
    """Exact round(numerator / denominator), with ties going to even."""
    quotient, remainder = divmod(int(numerator), int(denominator))
    twice = 2 * remainder
    if twice < denominator:
        return quotient
    if twice > denominator:
        return quotient + 1
    return quotient + (quotient % 2)


@dataclass(frozen=True)
class OnlinePRCKey:
    """Compact key/configuration for every causal prefix."""

    check_weight: int
    noise_rate: float
    support_key: bytes
    otp_key: bytes
    row_rate_numerator: int = DEFAULT_ROW_RATE_NUMERATOR
    row_rate_denominator: int = DEFAULT_ROW_RATE_DENOMINATOR
    scheme: str = SCHEME
    schedule_version: str = SCHEDULE_VERSION
    support_sampler_version: str = SUPPORT_SAMPLER_VERSION
    schema_version: int = KEY_SCHEMA_VERSION

    def __post_init__(self):
        if self.scheme != SCHEME:
            raise ValueError(f"unsupported scheme {self.scheme!r}")
        if self.schedule_version != SCHEDULE_VERSION:
            raise ValueError(
                f"unsupported schedule version {self.schedule_version!r}"
            )
        if self.support_sampler_version != SUPPORT_SAMPLER_VERSION:
            raise ValueError(
                "unsupported support sampler version "
                f"{self.support_sampler_version!r}"
            )
        if int(self.schema_version) != KEY_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported key schema version {self.schema_version!r}"
            )
        if int(self.check_weight) < 2:
            raise ValueError("check_weight must be at least 2")
        if not 0.0 <= float(self.noise_rate) < 0.5:
            raise ValueError("noise_rate must be in [0, 0.5)")
        if int(self.row_rate_denominator) <= 0:
            raise ValueError("row_rate_denominator must be positive")
        if not 0 < int(self.row_rate_numerator) <= int(
            self.row_rate_denominator
        ):
            raise ValueError("row rate must be in (0, 1]")
        if not isinstance(self.support_key, bytes) or not self.support_key:
            raise ValueError("support_key must be nonempty bytes")
        if not isinstance(self.otp_key, bytes) or not self.otp_key:
            raise ValueError("otp_key must be nonempty bytes")

    @classmethod
    def from_seed(
        cls,
        seed: int | bytes | bytearray | str,
        *,
        check_weight: int,
        noise_rate: float,
        row_rate_numerator: int = DEFAULT_ROW_RATE_NUMERATOR,
        row_rate_denominator: int = DEFAULT_ROW_RATE_DENOMINATOR,
    ) -> "OnlinePRCKey":
        root = _seed_key(seed)
        support_key = hmac.new(root, b"online-prc/support/v1", hashlib.sha256).digest()
        otp_key = hmac.new(root, b"online-prc/otp/v1", hashlib.sha256).digest()
        return cls(
            check_weight=int(check_weight),
            noise_rate=float(noise_rate),
            support_key=support_key,
            otp_key=otp_key,
            row_rate_numerator=int(row_rate_numerator),
            row_rate_denominator=int(row_rate_denominator),
        )

    def to_dict(self) -> dict:
        return {
            "check_weight": int(self.check_weight),
            "noise_rate": float(self.noise_rate),
            "support_key_hex": self.support_key.hex(),
            "otp_key_hex": self.otp_key.hex(),
            "row_rate_numerator": int(self.row_rate_numerator),
            "row_rate_denominator": int(self.row_rate_denominator),
            "scheme": self.scheme,
            "schedule_version": self.schedule_version,
            "support_sampler_version": self.support_sampler_version,
            "schema_version": int(self.schema_version),
        }

    @classmethod
    def from_dict(cls, value: dict) -> "OnlinePRCKey":
        return cls(
            check_weight=int(value["check_weight"]),
            noise_rate=float(value["noise_rate"]),
            support_key=bytes.fromhex(value["support_key_hex"]),
            otp_key=bytes.fromhex(value["otp_key_hex"]),
            row_rate_numerator=int(value["row_rate_numerator"]),
            row_rate_denominator=int(value["row_rate_denominator"]),
            scheme=value["scheme"],
            schedule_version=value["schedule_version"],
            support_sampler_version=value["support_sampler_version"],
            schema_version=int(value["schema_version"]),
        )

    @property
    def fingerprint(self) -> str:
        payload = repr(sorted(self.to_dict().items())).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def target_row_count(length: int, key: OnlinePRCKey) -> int:
    """Number of checks available at a realized prefix length."""
    length = int(length)
    if length < 0:
        raise ValueError("length must be nonnegative")
    requested = _round_rational_half_even(
        key.row_rate_numerator * length, key.row_rate_denominator
    )
    causal_maximum = max(0, length - (key.check_weight - 1))
    return min(requested, causal_maximum)


def is_parity_coordinate(position: int, key: OnlinePRCKey) -> bool:
    """Whether zero-based ``position`` introduces a new parity row."""
    position = int(position)
    if position < 0:
        raise ValueError("position must be nonnegative")
    return target_row_count(position + 1, key) > target_row_count(position, key)


def parent_indices(position: int, key: OnlinePRCKey) -> np.ndarray:
    """Deterministically select the distinct prior parents of a parity pivot."""
    position = int(position)
    if not is_parity_coordinate(position, key):
        raise ValueError(f"coordinate {position} is free, not a parity pivot")
    parent_count = key.check_weight - 1
    if position < parent_count:
        raise ValueError(
            f"coordinate {position} has only {position} predecessors; "
            f"need {parent_count}"
        )
    # Rejection sampling avoids modulo bias and, unlike delegating to NumPy's
    # choice implementation, fixes the support expansion byte-for-byte across
    # library versions.  Check weights are small, so duplicate retries are
    # inexpensive even at the first valid pivots.
    limit = (1 << 64) - ((1 << 64) % position)
    selected = set()
    draw = 0
    while len(selected) < parent_count:
        value = int.from_bytes(
            _expand(key.support_key, b"parents/v1", position, draw)[:8], "big"
        )
        draw += 1
        if value < limit:
            selected.add(value % position)
    return np.asarray(sorted(selected), dtype=np.int64)


def materialize_supports(length: int, key: OnlinePRCKey) -> np.ndarray:
    """Return the ``r(length) x check_weight`` causal support table."""
    rows = []
    for position in range(int(length)):
        if is_parity_coordinate(position, key):
            rows.append(np.append(parent_indices(position, key), position))
    if not rows:
        return np.empty((0, key.check_weight), dtype=np.int64)
    result = np.asarray(rows, dtype=np.int64)
    expected = target_row_count(length, key)
    if result.shape != (expected, key.check_weight):
        raise AssertionError(
            f"schedule produced shape {result.shape}, expected "
            f"{(expected, key.check_weight)}"
        )
    return result


def otp_bit(position: int, key: OnlinePRCKey) -> int:
    return _expand(key.otp_key, b"coordinate/v1", int(position))[0] & 1


def otp_prefix(length: int, key: OnlinePRCKey) -> np.ndarray:
    return np.fromiter(
        (otp_bit(i, key) for i in range(int(length))),
        dtype=np.uint8,
        count=int(length),
    )


def support_sha256(length: int, key: OnlinePRCKey) -> str:
    rows = materialize_supports(length, key)
    header = f"{rows.dtype}:{rows.shape}:".encode("utf-8")
    return hashlib.sha256(header + rows.tobytes()).hexdigest()


def parity_check_dense(length: int, key: OnlinePRCKey) -> np.ndarray:
    rows = materialize_supports(length, key)
    matrix = np.zeros((rows.shape[0], int(length)), dtype=np.uint8)
    if rows.size:
        matrix[np.arange(rows.shape[0])[:, None], rows] = 1
    return matrix


def reconstruct_generator(length: int, key: OnlinePRCKey) -> tuple[np.ndarray, np.ndarray]:
    """Post-hoc generator and free-coordinate list; never used in generation."""
    length = int(length)
    free = np.asarray(
        [i for i in range(length) if not is_parity_coordinate(i, key)],
        dtype=np.int64,
    )
    free_column = {int(position): column for column, position in enumerate(free)}
    generator = np.zeros((length, free.shape[0]), dtype=np.uint8)
    for position in range(length):
        if position in free_column:
            generator[position, free_column[position]] = 1
        else:
            generator[position] = np.bitwise_xor.reduce(
                generator[parent_indices(position, key)], axis=0
            )
    return generator, free


def gf2_rank(matrix: np.ndarray) -> int:
    array = np.asarray(matrix, dtype=np.uint8).copy() % 2
    rows, columns = array.shape
    rank = 0
    for column in range(columns):
        candidates = np.flatnonzero(array[rank:, column])
        if candidates.size == 0:
            continue
        pivot = rank + int(candidates[0])
        if pivot != rank:
            array[[rank, pivot]] = array[[pivot, rank]]
        for row in np.flatnonzero(array[:, column]):
            if row != rank:
                array[row] ^= array[rank]
        rank += 1
        if rank == rows:
            break
    return rank


def derive_document_seed(seed: int | bytes | bytearray | str, document_id: int) -> int:
    """Stable, batch-order-independent seed for one generated document."""
    digest = _expand(_seed_key(seed), b"online-prc/document/v1", int(document_id))
    return int.from_bytes(digest[:16], "big")


def _document_digest(document_seed: int, domain: bytes, position: int) -> bytes:
    return _expand(_seed_key(document_seed), domain, int(position))


def document_uniform(document_seed: int, domain: bytes | str,
                     position: int) -> float:
    """Return a deterministic uniform variate addressed by document/position.

    The open interval avoids the two inverse-CDF boundary cases.  Separate
    domains must be used for logically independent draws (for example bucket
    selection and token selection).  This is what makes online generation
    reproducible across batch order and resumable without serializing a global
    PyTorch RNG state.
    """
    if isinstance(domain, str):
        domain = domain.encode("utf-8")
    if not isinstance(domain, bytes) or not domain:
        raise ValueError("domain must be nonempty bytes or str")
    if int(document_seed) < 0 or int(position) < 0:
        raise ValueError("document_seed and position must be nonnegative")
    value = int.from_bytes(
        _document_digest(int(document_seed), domain, int(position))[:8],
        "big",
    )
    return (value + 0.5) / float(1 << 64)


class OnlinePRCEncoder:
    """Incremental encoder with independent state for every batch member."""

    def __init__(self, key: OnlinePRCKey, document_seeds: Sequence[int]):
        if not document_seeds:
            raise ValueError("document_seeds must be nonempty")
        seeds = [int(seed) for seed in document_seeds]
        if any(seed < 0 for seed in seeds):
            raise ValueError("document seeds must be nonnegative")
        if len(set(seeds)) != len(seeds):
            raise ValueError("document seeds must be unique within a batch")
        self.key = key
        self.document_seeds = tuple(seeds)
        self.clean_history: list[list[int]] = [[] for _ in seeds]
        self.error_history: list[list[int]] = [[] for _ in seeds]
        self.noisy_history: list[list[int]] = [[] for _ in seeds]

    @property
    def batch_size(self) -> int:
        return len(self.document_seeds)

    @property
    def lengths(self) -> np.ndarray:
        return np.asarray([len(row) for row in self.clean_history], dtype=np.int64)

    def next_bits(self, active: Iterable[bool] | None = None) -> np.ndarray:
        """Advance active documents once and return their noisy/OTP bits.

        Inactive entries return zero and are not advanced.  Because all free
        bits and noise are addressed by ``(document_seed, position)``, batch
        reordering or another document stopping cannot perturb a stream.
        """
        if active is None:
            active_array = np.ones(self.batch_size, dtype=bool)
        else:
            active_array = np.asarray(list(active), dtype=bool)
            if active_array.shape != (self.batch_size,):
                raise ValueError(
                    f"active mask shape {active_array.shape} does not match "
                    f"batch size {self.batch_size}"
                )

        output = np.zeros(self.batch_size, dtype=np.uint8)
        for row, enabled in enumerate(active_array):
            if not enabled:
                continue
            position = len(self.clean_history[row])
            if is_parity_coordinate(position, self.key):
                parents = parent_indices(position, self.key)
                clean = int(np.bitwise_xor.reduce(
                    np.asarray(self.clean_history[row], dtype=np.uint8)[parents]
                ))
            else:
                clean = _document_digest(
                    self.document_seeds[row], b"free-bit/v1", position
                )[0] & 1

            noise_word = int.from_bytes(
                _document_digest(
                    self.document_seeds[row], b"channel-noise/v1", position
                )[:8],
                "big",
            )
            error = int(noise_word < self.key.noise_rate * (1 << 64))
            noisy = clean ^ otp_bit(position, self.key) ^ error
            self.clean_history[row].append(clean)
            self.error_history[row].append(error)
            self.noisy_history[row].append(noisy)
            output[row] = noisy
        return output

    def encode_to_length(self, length: int) -> np.ndarray:
        length = int(length)
        if length < 0:
            raise ValueError("length must be nonnegative")
        if np.any(self.lengths > length):
            raise ValueError("encoder has already advanced beyond requested length")
        while np.any(self.lengths < length):
            self.next_bits(self.lengths < length)
        return np.asarray(self.noisy_history, dtype=np.uint8)

    def clean_array(self) -> np.ndarray:
        lengths = self.lengths
        if not np.all(lengths == lengths[0]):
            raise ValueError("batch members have different realized lengths")
        return np.asarray(self.clean_history, dtype=np.uint8)

    def error_array(self) -> np.ndarray:
        lengths = self.lengths
        if not np.all(lengths == lengths[0]):
            raise ValueError("batch members have different realized lengths")
        return np.asarray(self.error_history, dtype=np.uint8)


def validate_online_word(
    key: OnlinePRCKey,
    clean: Sequence[int],
    noisy: Sequence[int],
    error: Sequence[int],
) -> None:
    """Fail loudly unless the online algebra holds for this realized prefix."""
    clean_array = np.asarray(clean, dtype=np.uint8).reshape(-1)
    noisy_array = np.asarray(noisy, dtype=np.uint8).reshape(-1)
    error_array = np.asarray(error, dtype=np.uint8).reshape(-1)
    if not (clean_array.shape == noisy_array.shape == error_array.shape):
        raise ValueError("clean, noisy, and error arrays must have equal lengths")
    length = clean_array.shape[0]
    matrix = parity_check_dense(length, key)
    if np.any((matrix @ clean_array) % 2):
        raise AssertionError("clean online word violates a parity check")
    unpadded = noisy_array ^ otp_prefix(length, key)
    if not np.array_equal((matrix @ unpadded) % 2, (matrix @ error_array) % 2):
        raise AssertionError("noisy/OTP syndrome does not equal error syndrome")
