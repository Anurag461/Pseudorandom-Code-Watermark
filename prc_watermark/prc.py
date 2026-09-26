from __future__ import annotations
import numpy as np
import torch
import galois
from scipy.sparse import csr_matrix
from scipy.special import binom

GF = galois.GF(2)


def KeyGen(
    n,
    message_length=512,
    false_positive_rate=1e-09,
    t=3,
    g=None,
    r=None,
    noise_rate=None,
    seed=None,
):
    rng = np.random.default_rng(seed) if seed is not None else None

    def field_random(shape):
        return GF.Random(shape, seed=rng) if rng is not None else GF.Random(shape)

    def choice(*args, **kwargs):
        sampler = rng.choice if rng is not None else np.random.choice
        return sampler(*args, **kwargs)

    def permutation(*args, **kwargs):
        sampler = rng.permutation if rng is not None else np.random.permutation
        return sampler(*args, **kwargs)

    num_test_bits = int(np.ceil(np.log2(1 / false_positive_rate)))
    secpar = int(np.log2(binom(n, t)))
    if g is None:
        g = secpar
    if noise_rate is None:
        noise_rate = 1 - 2 ** (-secpar / g**2)
    k = message_length + g + num_test_bits
    if r is None:
        r = n - k - secpar
    generator_matrix = field_random((n, k))
    row_indices = []
    col_indices = []
    data = []
    for row in range(r):
        chosen_indices = choice(n - r + row, t - 1, replace=False)
        chosen_indices = np.append(chosen_indices, n - r + row)
        row_indices.extend([row] * t)
        col_indices.extend(chosen_indices)
        data.extend([1] * t)
        generator_matrix[n - r + row] = generator_matrix[chosen_indices[:-1]].sum(
            axis=0
        )
    parity_check_matrix = csr_matrix((data, (row_indices, col_indices)))
    max_bp_iter = int(np.log(n) / np.log(t))
    one_time_pad = field_random(n)
    test_bits = field_random(num_test_bits)
    bit_permutation = permutation(n)
    generator_matrix = generator_matrix[bit_permutation]
    one_time_pad = one_time_pad[bit_permutation]
    parity_check_matrix = parity_check_matrix[:, bit_permutation]
    encoding_key = (generator_matrix, one_time_pad, test_bits, g, noise_rate)
    decoding_key = (
        generator_matrix,
        parity_check_matrix,
        one_time_pad,
        false_positive_rate,
        noise_rate,
        test_bits,
        g,
        max_bp_iter,
        t,
    )
    return (encoding_key, decoding_key)


def Encode(encoding_key, message=None):
    generator_matrix, one_time_pad, test_bits, g, noise_rate = encoding_key
    n, k = generator_matrix.shape
    if message is None:
        payload = np.concatenate((test_bits, GF.Random(k - len(test_bits))))
    else:
        assert len(message) <= k - len(test_bits) - g, "Message is too long"
        payload = np.concatenate(
            (
                test_bits,
                GF.Random(g),
                GF(message),
                GF.Zeros(k - len(test_bits) - g - len(message)),
            )
        )
    error = GF(np.random.binomial(1, noise_rate, n))
    return 1 - 2 * torch.tensor(
        payload @ generator_matrix.T + one_time_pad + error, dtype=float
    )


def Detect(decoding_key, posteriors, false_positive_rate=None, return_info=False):
    (
        generator_matrix,
        parity_check_matrix,
        one_time_pad,
        false_positive_rate_key,
        noise_rate,
        test_bits,
        g,
        max_bp_iter,
        t,
    ) = decoding_key
    fpr = (
        false_positive_rate
        if false_positive_rate is not None
        else false_positive_rate_key
    )
    if torch is not None and isinstance(posteriors, torch.Tensor):
        S = posteriors.numpy(force=True).astype(np.float64)
    else:
        S = np.asarray(posteriors, dtype=np.float64)
    r = parity_check_matrix.shape[0]
    idx = parity_check_matrix.indices.reshape(r, t)
    S_w = np.prod(S[idx], axis=1)
    otp = np.asarray(one_time_pad, dtype=np.int64)
    a_w = np.prod(1 - 2 * otp[idx], axis=1).astype(np.float64)
    S_stat = float(np.sum(a_w * S_w))
    V = float(np.sum(S_w**2))
    tau = float(np.sqrt(2 * V * np.log(1 / fpr)))
    decision = bool(S_stat >= tau)
    if return_info:
        return (
            decision,
            {
                "method": "hoeffding",
                "statistic": S_stat,
                "threshold": tau,
                "V": V,
                "r": int(r),
                "fpr": float(fpr),
            },
        )
    return decision


from dataclasses import dataclass
import hashlib
import hmac
import struct
from typing import Iterable, Sequence

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
    quotient, remainder = divmod(int(numerator), int(denominator))
    twice = 2 * remainder
    if twice < denominator:
        return quotient
    if twice > denominator:
        return quotient + 1
    return quotient + quotient % 2


@dataclass(frozen=True)
class OnlinePRCKey:
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
            raise ValueError(f"unsupported schedule version {self.schedule_version!r}")
        if self.support_sampler_version != SUPPORT_SAMPLER_VERSION:
            raise ValueError(
                f"unsupported support sampler version {self.support_sampler_version!r}"
            )
        if int(self.schema_version) != KEY_SCHEMA_VERSION:
            raise ValueError(f"unsupported key schema version {self.schema_version!r}")
        if int(self.check_weight) < 2:
            raise ValueError("check_weight must be at least 2")
        if not 0.0 <= float(self.noise_rate) < 0.5:
            raise ValueError("noise_rate must be in [0, 0.5)")
        if int(self.row_rate_denominator) <= 0:
            raise ValueError("row_rate_denominator must be positive")
        if not 0 < int(self.row_rate_numerator) <= int(self.row_rate_denominator):
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
    length = int(length)
    if length < 0:
        raise ValueError("length must be nonnegative")
    requested = _round_rational_half_even(
        key.row_rate_numerator * length, key.row_rate_denominator
    )
    causal_maximum = max(0, length - (key.check_weight - 1))
    return min(requested, causal_maximum)


def is_parity_coordinate(position: int, key: OnlinePRCKey) -> bool:
    position = int(position)
    if position < 0:
        raise ValueError("position must be nonnegative")
    return target_row_count(position + 1, key) > target_row_count(position, key)


def parent_indices(position: int, key: OnlinePRCKey) -> np.ndarray:
    position = int(position)
    if not is_parity_coordinate(position, key):
        raise ValueError(f"coordinate {position} is free, not a parity pivot")
    parent_count = key.check_weight - 1
    if position < parent_count:
        raise ValueError(
            f"coordinate {position} has only {position} predecessors; need {parent_count}"
        )
    limit = (1 << 64) - (1 << 64) % position
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
            f"schedule produced shape {result.shape}, expected {(expected, key.check_weight)}"
        )
    return result


def otp_bit(position: int, key: OnlinePRCKey) -> int:
    return _expand(key.otp_key, b"coordinate/v1", int(position))[0] & 1


def otp_prefix(length: int, key: OnlinePRCKey) -> np.ndarray:
    return np.fromiter(
        (otp_bit(i, key) for i in range(int(length))), dtype=np.uint8, count=int(length)
    )


def derive_document_seed(seed: int | bytes | bytearray | str, document_id: int) -> int:
    digest = _expand(_seed_key(seed), b"online-prc/document/v1", int(document_id))
    return int.from_bytes(digest[:16], "big")


def _document_digest(document_seed: int, domain: bytes, position: int) -> bytes:
    return _expand(_seed_key(document_seed), domain, int(position))


def document_uniform(document_seed: int, domain: bytes | str, position: int) -> float:
    if isinstance(domain, str):
        domain = domain.encode("utf-8")
    if not isinstance(domain, bytes) or not domain:
        raise ValueError("domain must be nonempty bytes or str")
    if int(document_seed) < 0 or int(position) < 0:
        raise ValueError("document_seed and position must be nonnegative")
    value = int.from_bytes(
        _document_digest(int(document_seed), domain, int(position))[:8], "big"
    )
    return (value + 0.5) / float(1 << 64)


class OnlinePRCEncoder:

    def __init__(self, key: OnlinePRCKey, document_seeds: Sequence[int]):
        if not document_seeds:
            raise ValueError("document_seeds must be nonempty")
        seeds = [int(seed) for seed in document_seeds]
        if any((seed < 0 for seed in seeds)):
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
        if active is None:
            active_array = np.ones(self.batch_size, dtype=bool)
        else:
            active_array = np.asarray(list(active), dtype=bool)
            if active_array.shape != (self.batch_size,):
                raise ValueError(
                    f"active mask shape {active_array.shape} does not match batch size {self.batch_size}"
                )
        output = np.zeros(self.batch_size, dtype=np.uint8)
        for row, enabled in enumerate(active_array):
            if not enabled:
                continue
            position = len(self.clean_history[row])
            if is_parity_coordinate(position, self.key):
                parents = parent_indices(position, self.key)
                clean = int(
                    np.bitwise_xor.reduce(
                        np.asarray(self.clean_history[row], dtype=np.uint8)[parents]
                    )
                )
            else:
                clean = (
                    _document_digest(
                        self.document_seeds[row], b"free-bit/v1", position
                    )[0]
                    & 1
                )
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
