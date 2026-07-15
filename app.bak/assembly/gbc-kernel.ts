const PAGE_BYTES: usize = 64 * 1024;
const HEAP_START: usize = 1024 * 1024;
const BREED_COUNT: i32 = 49;

let cursor: usize = HEAP_START;

export function reserve(bytes: u32): usize {
  const pointer = (cursor + 15) & ~<usize>15;
  cursor = pointer + bytes;
  const requiredPages = (cursor + PAGE_BYTES - 1) / PAGE_BYTES;
  const currentPages = <usize>memory.size();
  if (requiredPages > currentPages) {
    memory.grow(<i32>(requiredPages - currentPages));
  }
  return pointer;
}

export function memoryBytes(): usize {
  return <usize>memory.size() * PAGE_BYTES;
}

export function accumulate(
  valuesPointer: usize,
  rowIndexesPointer: usize,
  genotypesPointer: usize,
  rowCount: i32,
  sampleCount: i32,
  gramPointer: usize,
  crossPointer: usize,
): void {
  for (let row = 0; row < rowCount; row += 1) {
    const referenceRow = load<u32>(rowIndexesPointer + <usize>row * 4);
    const frequenciesPointer =
      valuesPointer + <usize>referenceRow * <usize>BREED_COUNT * 8;
    const genotypePointer =
      genotypesPointer + <usize>row * <usize>sampleCount * 8;

    for (let breed = 0; breed < BREED_COUNT; breed += 1) {
      const frequency = load<f64>(frequenciesPointer + <usize>breed * 8);
      const frequencyVector = f64x2.splat(frequency);
      const crossRowPointer =
        crossPointer + <usize>breed * <usize>sampleCount * 8;

      let sample = 0;
      for (; sample + 1 < sampleCount; sample += 2) {
        const offset = <usize>sample * 8;
        const genotypes = v128.load(genotypePointer + offset);
        const current = v128.load(crossRowPointer + offset);
        v128.store(
          crossRowPointer + offset,
          f64x2.add(current, f64x2.mul(frequencyVector, genotypes)),
        );
      }
      if (sample < sampleCount) {
        const offset = <usize>sample * 8;
        store<f64>(
          crossRowPointer + offset,
          load<f64>(crossRowPointer + offset) +
            frequency * load<f64>(genotypePointer + offset),
        );
      }

      const gramRowPointer =
        gramPointer + <usize>breed * <usize>BREED_COUNT * 8;
      let otherBreed = breed;
      for (; otherBreed + 1 < BREED_COUNT; otherBreed += 2) {
        const frequencyOffset = <usize>otherBreed * 8;
        const gramOffset = <usize>otherBreed * 8;
        const frequencies = v128.load(frequenciesPointer + frequencyOffset);
        const current = v128.load(gramRowPointer + gramOffset);
        v128.store(
          gramRowPointer + gramOffset,
          f64x2.add(current, f64x2.mul(frequencyVector, frequencies)),
        );
      }
      if (otherBreed < BREED_COUNT) {
        const offset = <usize>otherBreed * 8;
        store<f64>(
          gramRowPointer + offset,
          load<f64>(gramRowPointer + offset) +
            frequency * load<f64>(frequenciesPointer + offset),
        );
      }
    }
  }
}
