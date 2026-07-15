export const BREED_COUNT = 49;
export const DEFAULT_BATCH_SIZE = 512;

const MISSING_VALUES = new Set([
  "-1.#IND",
  "1.#QNAN",
  "#N/A",
  "N/A",
  "NA",
  "#NA",
  "NULL",
  "NAN",
  "<NA>",
  "NONE",
  ".",
]);

export class GbcInputError extends Error {
  constructor(message) {
    super(message);
    this.name = "GbcInputError";
    this.code = "INPUT_ERROR";
  }
}

export class GbcLocalUnavailableError extends Error {
  constructor(message) {
    super(message);
    this.name = "GbcLocalUnavailableError";
    this.code = "LOCAL_UNAVAILABLE";
  }
}

function isWhitespace(code) {
  return code === 32 || code === 9 || code === 10 || code === 13;
}

function parseUnsignedInteger(text, start, end, lineNumber, label) {
  if (start >= end) {
    throw new GbcInputError(`Line ${lineNumber} has an invalid ${label}`);
  }
  let value = 0;
  for (let index = start; index < end; index += 1) {
    const digit = text.charCodeAt(index) - 48;
    if (digit < 0 || digit > 9) {
      throw new GbcInputError(`Line ${lineNumber} has an invalid ${label}`);
    }
    value = value * 10 + digit;
    if (!Number.isSafeInteger(value)) {
      throw new GbcInputError(`Line ${lineNumber} has an invalid ${label}`);
    }
  }
  return value;
}

function parseNumber(text, start, end) {
  let index = start;
  let sign = 1;
  const first = text.charCodeAt(index);
  if (first === 45 || first === 43) {
    sign = first === 45 ? -1 : 1;
    index += 1;
  }

  let value = 0;
  let digits = 0;
  while (index < end) {
    const digit = text.charCodeAt(index) - 48;
    if (digit < 0 || digit > 9) {
      break;
    }
    value = value * 10 + digit;
    digits += 1;
    index += 1;
  }

  if (index < end && text.charCodeAt(index) === 46) {
    index += 1;
    let place = 0.1;
    while (index < end) {
      const digit = text.charCodeAt(index) - 48;
      if (digit < 0 || digit > 9) {
        break;
      }
      value += digit * place;
      place *= 0.1;
      digits += 1;
      index += 1;
    }
  }

  if (digits === 0) {
    return Number.NaN;
  }
  if (index < end && (text.charCodeAt(index) === 69 || text.charCodeAt(index) === 101)) {
    index += 1;
    let exponentSign = 1;
    if (index < end && (text.charCodeAt(index) === 45 || text.charCodeAt(index) === 43)) {
      exponentSign = text.charCodeAt(index) === 45 ? -1 : 1;
      index += 1;
    }
    const exponentStart = index;
    let exponent = 0;
    while (index < end) {
      const digit = text.charCodeAt(index) - 48;
      if (digit < 0 || digit > 9) {
        break;
      }
      exponent = exponent * 10 + digit;
      index += 1;
    }
    if (index === exponentStart) {
      return Number.NaN;
    }
    value *= 10 ** (exponentSign * exponent);
  }
  return index === end ? sign * value : Number.NaN;
}

export function parseHeader(line, maxSamples) {
  const fields = line.replace(/^\uFEFF/, "").trim().split(/\s+/);
  if (fields.length < 2 || fields[0] !== "CHR:POS") {
    throw new GbcInputError(
      "The first row must start with CHR:POS and contain sample names"
    );
  }
  const sampleNames = fields.slice(1);
  if (sampleNames.length > maxSamples) {
    throw new GbcLocalUnavailableError(
      `The genotype file has ${sampleNames.length} samples; local analysis is limited to ${maxSamples}`
    );
  }
  return sampleNames;
}

export function parseDataLine(
  line,
  sampleCount,
  output,
  outputOffset,
  keyOutput,
  lineNumber
) {
  let index = 0;
  while (index < line.length && isWhitespace(line.charCodeAt(index))) {
    index += 1;
  }
  const keyStart = index;
  while (index < line.length && !isWhitespace(line.charCodeAt(index))) {
    index += 1;
  }
  const keyEnd = index;
  let colon = -1;
  for (let cursor = keyStart; cursor < keyEnd; cursor += 1) {
    if (line.charCodeAt(cursor) === 58) {
      colon = cursor;
      break;
    }
  }
  if (colon <= keyStart || colon >= keyEnd - 1) {
    throw new GbcInputError(
      `Line ${lineNumber} has an invalid CHR:POS identifier`
    );
  }
  keyOutput[0] = parseUnsignedInteger(
    line,
    keyStart,
    colon,
    lineNumber,
    "CHR:POS identifier"
  );
  keyOutput[1] = parseUnsignedInteger(
    line,
    colon + 1,
    keyEnd,
    lineNumber,
    "CHR:POS identifier"
  );
  if (keyOutput[0] > 0xffffffff || keyOutput[1] >= 1_000_000_000) {
    throw new GbcInputError(
      `Line ${lineNumber} has an out-of-range CHR:POS identifier`
    );
  }

  let values = 0;
  let missing = false;
  while (index < line.length) {
    while (index < line.length && isWhitespace(line.charCodeAt(index))) {
      index += 1;
    }
    if (index >= line.length) {
      break;
    }
    const tokenStart = index;
    while (index < line.length && !isWhitespace(line.charCodeAt(index))) {
      index += 1;
    }
    const tokenEnd = index;
    if (values >= sampleCount) {
      throw new GbcInputError(
        `Line ${lineNumber} has more than ${sampleCount} genotypes`
      );
    }
    const value = parseNumber(line, tokenStart, tokenEnd);
    if (Number.isFinite(value)) {
      output[outputOffset + values] = value;
    } else {
      const token = line.slice(tokenStart, tokenEnd).toUpperCase();
      if (!MISSING_VALUES.has(token)) {
        throw new GbcInputError(
          `Line ${lineNumber} contains an invalid genotype value`
        );
      }
      missing = true;
      output[outputOffset + values] = 0;
    }
    values += 1;
  }
  if (values !== sampleCount) {
    throw new GbcInputError(
      `Line ${lineNumber} has ${values} genotypes; expected ${sampleCount}`
    );
  }
  return !missing;
}

function compareKey(chromosome, position, otherChromosome, otherPosition) {
  if (chromosome !== otherChromosome) {
    return chromosome < otherChromosome ? -1 : 1;
  }
  if (position === otherPosition) {
    return 0;
  }
  return position < otherPosition ? -1 : 1;
}

export class ReferenceMatcher {
  constructor(keys, snps) {
    this.keys = keys;
    this.snps = snps;
    this.cursor = 0;
    this.previousChromosome = 0;
    this.previousPosition = 0;
    this.hasPrevious = false;
    this.monotonic = true;
  }

  find(chromosome, position) {
    if (
      this.hasPrevious &&
      compareKey(
        chromosome,
        position,
        this.previousChromosome,
        this.previousPosition
      ) < 0
    ) {
      this.monotonic = false;
    }
    this.hasPrevious = true;
    this.previousChromosome = chromosome;
    this.previousPosition = position;

    if (this.monotonic) {
      while (this.cursor < this.snps) {
        const offset = this.cursor * 2;
        const comparison = compareKey(
          this.keys[offset],
          this.keys[offset + 1],
          chromosome,
          position
        );
        if (comparison >= 0) {
          return comparison === 0 ? this.cursor : -1;
        }
        this.cursor += 1;
      }
      return -1;
    }

    let low = 0;
    let high = this.snps - 1;
    while (low <= high) {
      const middle = (low + high) >>> 1;
      const offset = middle * 2;
      const comparison = compareKey(
        this.keys[offset],
        this.keys[offset + 1],
        chromosome,
        position
      );
      if (comparison === 0) {
        return middle;
      }
      if (comparison < 0) {
        low = middle + 1;
      } else {
        high = middle - 1;
      }
    }
    return -1;
  }
}

export class GbcAccumulator {
  constructor(wasm, manifest, sampleCount, batchSize = DEFAULT_BATCH_SIZE) {
    if (manifest.breeds.length !== BREED_COUNT) {
      throw new GbcLocalUnavailableError(
        `The browser kernel expects ${BREED_COUNT} reference breeds`
      );
    }
    this.wasm = wasm;
    this.manifest = manifest;
    this.sampleCount = sampleCount;
    this.batchSize = batchSize;
    this.batchCount = 0;
    this.matchedSnps = 0;

    this.payloadPointer = wasm.reserve(manifest.uncompressedBytes);
    this.rowIndexesPointer = wasm.reserve(batchSize * 4);
    this.genotypesPointer = wasm.reserve(batchSize * sampleCount * 8);
    this.gramPointer = wasm.reserve(BREED_COUNT * BREED_COUNT * 8);
    this.crossPointer = wasm.reserve(BREED_COUNT * sampleCount * 8);

    const memory = wasm.memory.buffer;
    this.payload = new Uint8Array(
      memory,
      this.payloadPointer,
      manifest.uncompressedBytes
    );
    this.rowIndexes = new Uint32Array(
      memory,
      this.rowIndexesPointer,
      batchSize
    );
    this.genotypes = new Float64Array(
      memory,
      this.genotypesPointer,
      batchSize * sampleCount
    );
    this.gram = new Float64Array(
      memory,
      this.gramPointer,
      BREED_COUNT * BREED_COUNT
    );
    this.cross = new Float64Array(
      memory,
      this.crossPointer,
      BREED_COUNT * sampleCount
    );
    this.gram.fill(0);
    this.cross.fill(0);
  }

  initializeReference() {
    const header = new DataView(
      this.wasm.memory.buffer,
      this.payloadPointer,
      this.manifest.headerBytes
    );
    const magic = new TextDecoder().decode(
      this.payload.subarray(0, 8)
    );
    if (
      magic !== "CBITGBC1" ||
      header.getUint32(8, true) !== this.manifest.schemaVersion ||
      header.getUint32(12, true) !== this.manifest.snps ||
      header.getUint32(16, true) !== this.manifest.breeds.length
    ) {
      throw new GbcLocalUnavailableError("The browser GBC reference is invalid");
    }
    this.keys = new Uint32Array(
      this.wasm.memory.buffer,
      this.payloadPointer + this.manifest.keysOffset,
      this.manifest.snps * 2
    );
    this.valuesPointer = this.payloadPointer + this.manifest.valuesOffset;
    this.matcher = new ReferenceMatcher(this.keys, this.manifest.snps);
  }

  genotypeOffset() {
    return this.batchCount * this.sampleCount;
  }

  commit(referenceIndex) {
    this.rowIndexes[this.batchCount] = referenceIndex;
    this.batchCount += 1;
    this.matchedSnps += 1;
    if (this.batchCount === this.batchSize) {
      this.flush();
    }
  }

  flush() {
    if (this.batchCount === 0) {
      return;
    }
    this.wasm.accumulate(
      this.valuesPointer,
      this.rowIndexesPointer,
      this.genotypesPointer,
      this.batchCount,
      this.sampleCount,
      this.gramPointer,
      this.crossPointer
    );
    this.batchCount = 0;
  }

  finish() {
    this.flush();
    if (this.matchedSnps === 0) {
      throw new GbcInputError(
        "No SNPs in the genotype file match the reference panel"
      );
    }
    for (let row = 0; row < BREED_COUNT; row += 1) {
      for (let column = row + 1; column < BREED_COUNT; column += 1) {
        this.gram[column * BREED_COUNT + row] =
          this.gram[row * BREED_COUNT + column];
      }
    }
  }
}

export function solveLinearSystem(matrix, rightHandSide, size, columns) {
  const coefficients = new Float64Array(rightHandSide);
  const factors = new Float64Array(matrix);

  for (let pivotColumn = 0; pivotColumn < size; pivotColumn += 1) {
    let pivotRow = pivotColumn;
    let pivotValue = Math.abs(factors[pivotColumn * size + pivotColumn]);
    for (let row = pivotColumn + 1; row < size; row += 1) {
      const candidate = Math.abs(factors[row * size + pivotColumn]);
      if (candidate > pivotValue) {
        pivotValue = candidate;
        pivotRow = row;
      }
    }
    if (!Number.isFinite(pivotValue) || pivotValue <= Number.EPSILON) {
      throw new GbcLocalUnavailableError(
        "The local GBC matrix is singular; server fallback is required"
      );
    }
    if (pivotRow !== pivotColumn) {
      for (let column = 0; column < size; column += 1) {
        const first = pivotColumn * size + column;
        const second = pivotRow * size + column;
        const temporary = factors[first];
        factors[first] = factors[second];
        factors[second] = temporary;
      }
      for (let column = 0; column < columns; column += 1) {
        const first = pivotColumn * columns + column;
        const second = pivotRow * columns + column;
        const temporary = coefficients[first];
        coefficients[first] = coefficients[second];
        coefficients[second] = temporary;
      }
    }

    const diagonal = factors[pivotColumn * size + pivotColumn];
    for (let row = pivotColumn + 1; row < size; row += 1) {
      const rowOffset = row * size;
      const factor = factors[rowOffset + pivotColumn] / diagonal;
      factors[rowOffset + pivotColumn] = factor;
      for (let column = pivotColumn + 1; column < size; column += 1) {
        factors[rowOffset + column] -=
          factor * factors[pivotColumn * size + column];
      }
      const resultOffset = row * columns;
      const pivotResultOffset = pivotColumn * columns;
      for (let column = 0; column < columns; column += 1) {
        coefficients[resultOffset + column] -=
          factor * coefficients[pivotResultOffset + column];
      }
    }
  }

  for (let row = size - 1; row >= 0; row -= 1) {
    const diagonal = factors[row * size + row];
    const resultOffset = row * columns;
    for (let column = 0; column < columns; column += 1) {
      let value = coefficients[resultOffset + column];
      for (let following = row + 1; following < size; following += 1) {
        value -=
          factors[row * size + following] *
          coefficients[following * columns + column];
      }
      coefficients[resultOffset + column] = value / diagonal;
    }
  }
  return coefficients;
}

function roundFour(value) {
  const scaled = value * 10_000;
  const lower = Math.floor(scaled);
  const fraction = scaled - lower;
  if (fraction > 0.5 || (fraction === 0.5 && lower % 2 !== 0)) {
    return (lower + 1) / 10_000;
  }
  return lower / 10_000;
}

export function normalise(coefficients, threshold, breeds, sampleNames) {
  const samples = sampleNames.length;
  const values = new Float64Array(coefficients.length);
  for (let sample = 0; sample < samples; sample += 1) {
    let total = 0;
    for (let breed = 0; breed < breeds.length; breed += 1) {
      const offset = breed * samples + sample;
      const value = Math.max(0, coefficients[offset]);
      values[offset] = value;
      total += value;
    }
    let retainedTotal = 0;
    for (let breed = 0; breed < breeds.length; breed += 1) {
      const offset = breed * samples + sample;
      const contribution = values[offset] / total;
      values[offset] = contribution < threshold ? 0 : contribution;
      retainedTotal += values[offset];
    }
    for (let breed = 0; breed < breeds.length; breed += 1) {
      const offset = breed * samples + sample;
      values[offset] = roundFour(values[offset] / retainedTotal);
    }
  }
  return values;
}

function escapeCsv(value) {
  const text = String(value);
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

export function serialiseResult(breeds, sampleNames, values) {
  const data = [];
  const csv = [["", ...sampleNames].map(escapeCsv).join(",")];
  for (let breed = 0; breed < breeds.length; breed += 1) {
    const record = { "Unnamed: 0": breeds[breed] };
    const row = [breeds[breed]];
    for (let sample = 0; sample < sampleNames.length; sample += 1) {
      const value = values[breed * sampleNames.length + sample];
      const serialised = Number.isFinite(value) ? value : null;
      record[sampleNames[sample]] = serialised;
      row.push(serialised === null ? "" : serialised);
    }
    data.push(record);
    csv.push(row.map(escapeCsv).join(","));
  }
  return {
    columns: ["Unnamed: 0", ...sampleNames],
    data,
    csv: `${csv.join("\n")}\n`,
  };
}
