import { describe, expect, it } from "vitest";

import { createPcm16Encoder, PCM_CHUNK_SAMPLES } from "@/lib/pcm-capture";

describe("createPcm16Encoder", () => {
  it("downsamples 48 kHz audio to 16 kHz PCM16 chunks and flushes the remainder", () => {
    const chunks: Int16Array[] = [];
    const encoder = createPcm16Encoder(48_000, (pcm) => chunks.push(pcm));
    const input = new Float32Array(PCM_CHUNK_SAMPLES * 3 + 30);
    for (let index = 0; index < input.length; index += 3) {
      input.set([0.4, 0.5, 0.6], index);
    }

    encoder.push(input.subarray(0, 4_000));
    encoder.push(input.subarray(4_000));
    encoder.flush();

    expect(chunks.map((chunk) => chunk.length)).toEqual([PCM_CHUNK_SAMPLES, 10]);
    expect(chunks.every((chunk) => chunk.every((sample) => sample === Math.trunc(0.5 * 0x7fff)))).toBe(true);
  });
});
