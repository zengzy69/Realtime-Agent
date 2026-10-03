/** Gapless Web Audio playback of streamed mono PCM16 chunks. */
export interface PcmPlayer {
  push: (pcm: Int16Array) => void;
  /** Resolves once everything pushed so far has played. */
  drain: () => Promise<void>;
  stop: () => void;
}

const START_DELAY_S = 0.05;

export function createPcmPlayer(sampleRate: number): PcmPlayer {
  const context = new AudioContext();
  void context.resume();
  const sources = new Set<AudioBufferSourceNode>();
  let nextStart = 0;
  let stopped = false;

  return {
    push: (pcm) => {
      if (stopped || pcm.length === 0) return;
      const buffer = context.createBuffer(1, pcm.length, sampleRate);
      const channel = buffer.getChannelData(0);
      for (let index = 0; index < pcm.length; index++) channel[index] = pcm[index] / 32768;
      const source = context.createBufferSource();
      source.buffer = buffer;
      source.connect(context.destination);
      source.onended = () => sources.delete(source);
      nextStart = Math.max(nextStart, context.currentTime + START_DELAY_S);
      source.start(nextStart);
      nextStart += buffer.duration;
      sources.add(source);
    },
    drain: () => {
      if (stopped) return Promise.resolve();
      const remainingMs = Math.max(0, (nextStart - context.currentTime) * 1000);
      return new Promise((resolve) => setTimeout(resolve, remainingMs));
    },
    stop: () => {
      if (stopped) return;
      stopped = true;
      for (const source of sources) source.stop();
      sources.clear();
      void context.close();
    },
  };
}
