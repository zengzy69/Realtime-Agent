/** Capture microphone audio as 16 kHz mono PCM16 chunks for realtime transcription. */

export const PCM_SAMPLE_RATE = 16_000;
/** 200 ms per chunk, the packet size Doubao recommends for streaming ASR. */
export const PCM_CHUNK_SAMPLES = 3_200;

const PCM_WORKLET_NAME = "nanobot-pcm-capture";
const PCM_WORKLET_SOURCE = `
class NanobotPcmCapture extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (channel) this.port.postMessage(channel.slice(0));
    return true;
  }
}
registerProcessor("${PCM_WORKLET_NAME}", NanobotPcmCapture);
`;

export interface PcmCapture {
  /** Emit any buffered samples, then release the audio graph. */
  stop: () => void;
}

export interface Pcm16Encoder {
  push: (samples: Float32Array) => void;
  flush: () => void;
}

/**
 * Downsample float audio to 16 kHz PCM16 by averaging each output sample's
 * input span, and emit fixed-size chunks.
 */
export function createPcm16Encoder(
  inputSampleRate: number,
  onChunk: (pcm: Int16Array) => void,
): Pcm16Encoder {
  const step = inputSampleRate / PCM_SAMPLE_RATE;
  let chunk = new Int16Array(PCM_CHUNK_SAMPLES);
  let filled = 0;
  let sum = 0;
  let count = 0;
  let untilNext = step;

  const emit = (value: number) => {
    const sample = Math.max(-1, Math.min(1, value));
    chunk[filled] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
    filled += 1;
    if (filled === PCM_CHUNK_SAMPLES) {
      onChunk(chunk);
      chunk = new Int16Array(PCM_CHUNK_SAMPLES);
      filled = 0;
    }
  };

  return {
    push(samples) {
      for (const sample of samples) {
        sum += sample;
        count += 1;
        untilNext -= 1;
        if (untilNext > 0) continue;
        const average = sum / count;
        sum = 0;
        count = 0;
        while (untilNext <= 0) {
          emit(average);
          untilNext += step;
        }
      }
    },
    flush() {
      if (filled === 0) return;
      onChunk(chunk.slice(0, filled));
      chunk = new Int16Array(PCM_CHUNK_SAMPLES);
      filled = 0;
    },
  };
}

export async function startPcmCapture(
  stream: MediaStream,
  onChunk: (pcm: Int16Array) => void,
): Promise<PcmCapture> {
  const AudioContextCtor = window.AudioContext
    ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (!AudioContextCtor || typeof AudioWorkletNode === "undefined") {
    throw new DOMException("AudioWorklet is unavailable", "NotSupportedError");
  }
  const context = new AudioContextCtor();
  try {
    const moduleUrl = URL.createObjectURL(
      new Blob([PCM_WORKLET_SOURCE], { type: "application/javascript" }),
    );
    try {
      await context.audioWorklet.addModule(moduleUrl);
    } finally {
      URL.revokeObjectURL(moduleUrl);
    }
    const source = context.createMediaStreamSource(stream);
    const node = new AudioWorkletNode(context, PCM_WORKLET_NAME);
    const encoder = createPcm16Encoder(context.sampleRate, onChunk);
    node.port.onmessage = (event: MessageEvent<Float32Array>) => encoder.push(event.data);
    source.connect(node);
    node.connect(context.destination);
    void context.resume().catch(() => undefined);
    return {
      stop() {
        node.port.onmessage = null;
        source.disconnect();
        node.disconnect();
        encoder.flush();
        void context.close().catch(() => undefined);
      },
    };
  } catch (error) {
    void context.close().catch(() => undefined);
    throw error;
  }
}
