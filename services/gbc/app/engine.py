import csv
import io
import re
import time
from dataclasses import dataclass
from typing import BinaryIO, Iterable

import numpy as np

from .reference import ReferenceData


KEY_MULTIPLIER = 1_000_000_000
MISSING_VALUE = re.compile(
    rb"(?:^|\s)(?:-?1\.\#(?:IND|QNAN)|\#N/A|N/A|NA|\#NA|"
    rb"NULL|-?NAN|<NA>|NONE|\.)(?=\s|$)",
    re.IGNORECASE,
)
LINE_BREAK = re.compile(rb"\r\n?|\n")


class GbcInputError(ValueError):
    pass


@dataclass(frozen=True)
class GbcResult:
    columns: list[str]
    data: list[dict[str, object]]
    csv: str
    matched_snps: int
    samples: int
    elapsed_seconds: float


def _iter_lines(stream: BinaryIO) -> Iterable[bytes]:
    remainder = b""
    while True:
        chunk = stream.read(1024 * 1024)
        if not chunk:
            break
        parts = LINE_BREAK.split(remainder + chunk)
        remainder = parts.pop()
        yield from parts
    if remainder:
        yield remainder


def _encode_key(value: bytes, line_number: int) -> int:
    try:
        chromosome, position = value.split(b":", 1)
        chromosome_number = int(chromosome)
        position_number = int(position)
    except (TypeError, ValueError) as error:
        raise GbcInputError(
            f"Line {line_number} has an invalid CHR:POS identifier"
        ) from error
    if chromosome_number < 0 or position_number < 0 or position_number >= KEY_MULTIPLIER:
        raise GbcInputError(
            f"Line {line_number} has an out-of-range CHR:POS identifier"
        )
    return chromosome_number * KEY_MULTIPLIER + position_number


def _accumulate(
    reference: ReferenceData,
    batch_keys: np.ndarray,
    batch_genotypes: np.ndarray,
    count: int,
    gram: np.ndarray,
    cross: np.ndarray,
) -> int:
    keys = batch_keys[:count]
    locations = np.searchsorted(reference.keys, keys)
    found = locations < len(reference.keys)
    valid_locations = locations[found]
    found[found] = reference.keys[valid_locations] == keys[found]
    if not np.any(found):
        return 0

    allele_frequencies = np.asarray(reference.values[locations[found]])
    genotypes = batch_genotypes[:count][found]
    gram += allele_frequencies.T @ allele_frequencies
    cross += allele_frequencies.T @ genotypes
    return int(np.count_nonzero(found))


def _normalise(coefficients: np.ndarray, threshold: float) -> np.ndarray:
    coefficients = coefficients.copy()
    coefficients[coefficients < 0] = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        contributions = coefficients / coefficients.sum(axis=0, keepdims=True)
        contributions[contributions < threshold] = 0
        contributions = contributions / contributions.sum(axis=0, keepdims=True)
    return np.round(contributions, 4)


def _serialise(
    breeds: tuple[str, ...], sample_names: list[str], values: np.ndarray
) -> tuple[list[dict[str, object]], str]:
    records = []
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow([""] + sample_names)

    for breed_index, breed in enumerate(breeds):
        row = values[breed_index]
        serialised = [None if not np.isfinite(value) else float(value) for value in row]
        record = {"Unnamed: 0": breed}
        record.update(zip(sample_names, serialised))
        records.append(record)
        writer.writerow([breed] + ["" if value is None else value for value in serialised])
    return records, output.getvalue()


class GbcEngine:
    def __init__(
        self,
        reference: ReferenceData,
        batch_size: int = 2048,
        max_samples: int = 1000,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if max_samples < 1:
            raise ValueError("max_samples must be positive")
        self.reference = reference
        self.batch_size = batch_size
        self.max_samples = max_samples

    def estimate(self, stream: BinaryIO, threshold: float) -> GbcResult:
        started = time.perf_counter()
        lines = iter(_iter_lines(stream))
        header = None
        for raw_line in lines:
            if raw_line.strip():
                header = raw_line.decode("utf-8-sig").split()
                break
        if not header or header[0] != "CHR:POS" or len(header) < 2:
            raise GbcInputError(
                "The first row must start with CHR:POS and contain sample names"
            )

        sample_names = header[1:]
        sample_count = len(sample_names)
        if sample_count > self.max_samples:
            raise GbcInputError(
                f"The genotype file has {sample_count} samples; "
                f"the limit is {self.max_samples}"
            )
        breed_count = len(self.reference.breeds)
        batch_keys = np.empty(self.batch_size, dtype=np.uint64)
        batch_genotypes = np.empty(
            (self.batch_size, sample_count), dtype=np.float64
        )
        gram = np.zeros((breed_count, breed_count), dtype=np.float64)
        cross = np.zeros((breed_count, sample_count), dtype=np.float64)
        batch_count = 0
        matched_snps = 0
        line_number = 1

        for raw_line in lines:
            line_number += 1
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                raw_key, raw_values = raw_line.split(None, 1)
            except ValueError as error:
                raise GbcInputError(
                    f"Line {line_number} does not contain genotype values"
                ) from error

            # The original pandas implementation drops a SNP if any sample is missing.
            if MISSING_VALUE.search(raw_values):
                continue
            try:
                genotypes = np.fromstring(
                    raw_values.decode("ascii"), sep=" ", dtype=np.float64
                )
            except UnicodeDecodeError as error:
                raise GbcInputError(
                    f"Line {line_number} contains non-ASCII genotype values"
                ) from error
            if genotypes.size != sample_count:
                raise GbcInputError(
                    f"Line {line_number} has {genotypes.size} genotypes; expected {sample_count}"
                )

            batch_keys[batch_count] = _encode_key(raw_key, line_number)
            batch_genotypes[batch_count] = genotypes
            batch_count += 1
            if batch_count == self.batch_size:
                matched_snps += _accumulate(
                    self.reference,
                    batch_keys,
                    batch_genotypes,
                    batch_count,
                    gram,
                    cross,
                )
                batch_count = 0

        if batch_count:
            matched_snps += _accumulate(
                self.reference,
                batch_keys,
                batch_genotypes,
                batch_count,
                gram,
                cross,
            )
        if matched_snps == 0:
            raise GbcInputError("No SNPs in the genotype file match the reference panel")

        try:
            coefficients = np.linalg.solve(gram, cross)
        except np.linalg.LinAlgError:
            coefficients = np.linalg.lstsq(gram, cross, rcond=None)[0]
        contributions = _normalise(coefficients, threshold)
        records, csv_text = _serialise(
            self.reference.breeds, sample_names, contributions
        )
        return GbcResult(
            columns=["Unnamed: 0"] + sample_names,
            data=records,
            csv=csv_text,
            matched_snps=matched_snps,
            samples=sample_count,
            elapsed_seconds=time.perf_counter() - started,
        )
