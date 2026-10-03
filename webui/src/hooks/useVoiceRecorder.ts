import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
} from "react";

import { startPcmCapture, type PcmCapture } from "@/lib/pcm-capture";
import type {
  RealtimeTranscription,
  StartRealtimeTranscription,
  TranscribeAudioOptions,
} from "@/lib/types";

const VOICE_RECORDING_MAX_MS = 120_000;
const VOICE_RECORDING_MIN_MS = 650;
/** Hands-free listen after a spoken reply: close the mic if nothing was heard. */
const VOICE_AUTO_LISTEN_IDLE_MS = 8_000;
/** Delay between the end of one live transcription request and the next. */
const VOICE_LIVE_INTERVAL_MS = 500;
const VOICE_LIVE_TIMESLICE_MS = 250;
/** ASR providers resample to 16 kHz; sending more only grows each upload. */
const VOICE_WAV_SAMPLE_RATE = 16_000;
const VOICE_HOLD_START_MS = 140;
const VOICE_WAVEFORM_BAR_COUNT = 64;
const VOICE_WAVEFORM_SILENT_HEIGHT = 3;
const VOICE_WAVEFORM_MIN_HEIGHT = 7;
const VOICE_WAVEFORM_MAX_HEIGHT = 34;
const VOICE_MIN_LEVEL = 0.018;
const VOICE_WAVEFORM_IDLE_LEVELS = Array.from(
  { length: VOICE_WAVEFORM_BAR_COUNT },
  () => VOICE_WAVEFORM_SILENT_HEIGHT,
);
const VOICE_MIME_CANDIDATES = [
  "audio/webm;codecs=opus",
  "audio/webm",
  "audio/mp4",
  "audio/ogg;codecs=opus",
] as const;

export type VoiceRecorderState = "idle" | "recording" | "transcribing";
export type VoiceRecorderErrorKey =
  | "failed"
  | "insecureContext"
  | "noDevice"
  | "notConfigured"
  | "permission"
  | "tooLong"
  | "tooShort"
  | "unsupported";

interface VoiceRecorderOptions {
  disabled?: boolean;
  onClearError: () => void;
  onError: (key: VoiceRecorderErrorKey) => void;
  onTranscript: (text: string) => void;
  /** Partial text reported before `onTranscript` by streaming providers. */
  onTranscriptDelta?: (delta: string) => void;
  onTranscribeAudio?: (dataUrl: string, options?: TranscribeAudioOptions) => Promise<string>;
  /** When true, convert recorded audio to WAV before sending (needed for providers that don't support WebM). */
  wantsWav?: boolean;
  /** Re-transcribe the audio captured so far while recording continues. */
  live?: boolean;
  /** Full transcript of the audio captured so far; superseded by `onTranscript`. */
  onInterimTranscript?: (text: string) => void;
  /** Stream microphone PCM to a realtime provider instead of recording a clip. */
  realtime?: boolean;
  onStartRealtimeTranscription?: StartRealtimeTranscription;
}

interface RealtimeRecording {
  capture: PcmCapture;
  transcription: RealtimeTranscription;
}

export function useVoiceRecorder({
  disabled,
  onClearError,
  onError,
  onTranscript,
  onTranscriptDelta,
  onTranscribeAudio,
  wantsWav = false,
  live = false,
  onInterimTranscript,
  realtime = false,
  onStartRealtimeTranscription,
}: VoiceRecorderOptions) {
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const realtimeRef = useRef<RealtimeRecording | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const audioRef = useRef<VoiceAudioState | null>(null);
  const startedAtRef = useRef(0);
  const maxTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const liveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  /** Incremented when a recording ends so late live results are discarded. */
  const recordingSessionRef = useRef(0);
  const holdTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const holdActiveRef = useRef(false);
  const startPendingRef = useRef(false);
  const stopAfterStartRef = useRef(false);
  const suppressClickRef = useRef(false);
  const suppressClickTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const shortcutActiveRef = useRef(false);
  const autoListenRef = useRef(false);
  const heardSpeechRef = useRef(false);
  const idleTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [state, setState] = useState<VoiceRecorderState>("idle");
  const [elapsedMs, setElapsedMs] = useState(0);
  const [levels, setLevels] = useState<number[]>(VOICE_WAVEFORM_IDLE_LEVELS);

  const clearSuppressClickTimer = useCallback(() => clearTimer(suppressClickTimerRef), []);

  const suppressNextClick = useCallback(() => {
    clearSuppressClickTimer();
    suppressClickRef.current = true;
    suppressClickTimerRef.current = setTimeout(() => {
      suppressClickRef.current = false;
      suppressClickTimerRef.current = null;
    }, 500);
  }, [clearSuppressClickTimer]);

  const stopWaveform = useCallback(() => {
    const audio = audioRef.current;
    audioRef.current = null;
    if (!audio) return;
    if (audio.frame !== null) cancelAnimationFrame(audio.frame);
    audio.source.disconnect();
    audio.analyser.disconnect();
    void audio.context.close().catch(() => undefined);
  }, []);

  const startWaveform = useCallback((stream: MediaStream) => {
    const AudioContextCtor = audioContextConstructor();
    if (!AudioContextCtor) return;
    stopWaveform();
    setLevels(VOICE_WAVEFORM_IDLE_LEVELS);
    try {
      const context = new AudioContextCtor();
      const source = context.createMediaStreamSource(stream);
      const analyser = context.createAnalyser();
      analyser.fftSize = 256;
      analyser.smoothingTimeConstant = 0.68;
      source.connect(analyser);
      const audio: VoiceAudioState = {
        analyser,
        context,
        data: new Uint8Array(analyser.fftSize),
        frame: null,
        source,
      };
      const tick = () => {
        const current = audioRef.current;
        if (!current) return;
        if (current.context.state !== "running") {
          void current.context.resume().catch(() => undefined);
          current.frame = requestAnimationFrame(tick);
          return;
        }
        current.analyser.getByteTimeDomainData(current.data);
        const level = voiceLevelFromSamples(current.data);
        setLevels((currentLevels) => [
          ...currentLevels.slice(1),
          waveformHeightFromLevel(level),
        ]);
        current.frame = requestAnimationFrame(tick);
      };
      audioRef.current = audio;
      void context.resume().catch(() => undefined);
      audio.frame = requestAnimationFrame(tick);
    } catch {
      stopWaveform();
    }
  }, [stopWaveform]);

  const cleanupRecording = useCallback(() => {
    recordingSessionRef.current += 1;
    clearTimer(holdTimerRef);
    clearTimer(maxTimerRef);
    clearTimer(liveTimerRef);
    clearTimer(idleTimerRef);
    stopWaveform();
    const realtimeRecording = realtimeRef.current;
    realtimeRef.current = null;
    if (realtimeRecording) {
      realtimeRecording.capture.stop();
      realtimeRecording.transcription.cancel();
    }
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    mediaRecorderRef.current = null;
    startPendingRef.current = false;
    shortcutActiveRef.current = false;
  }, [stopWaveform]);

  const finishRealtimeRecording = useCallback(() => {
    const realtimeRecording = realtimeRef.current;
    if (!realtimeRecording) return;
    const durationMs = Math.max(0, Date.now() - startedAtRef.current);
    if (durationMs < VOICE_RECORDING_MIN_MS) {
      const auto = autoListenRef.current;
      realtimeRecording.transcription.cancel();
      realtimeRef.current = null;
      realtimeRecording.capture.stop();
      autoListenRef.current = false;
      cleanupRecording();
      setState("idle");
      if (!auto) onError("tooShort");
      return;
    }
    realtimeRecording.capture.stop();
    realtimeRecording.transcription.finish();
  }, [cleanupRecording, onError]);

  const stopRecording = useCallback(() => {
    if (realtimeRef.current) {
      finishRealtimeRecording();
      return;
    }
    const recorder = mediaRecorderRef.current;
    if (!recorder || recorder.state === "inactive") return;
    recorder.stop();
  }, [finishRealtimeRecording]);

  const stopRecordingWhenReady = useCallback(() => {
    const recorder = mediaRecorderRef.current;
    if (realtimeRef.current || (recorder && recorder.state !== "inactive")) {
      stopRecording();
    } else if (startPendingRef.current) {
      stopAfterStartRef.current = true;
    }
  }, [stopRecording]);

  const startRealtimeRecording = useCallback(async (
    stream: MediaStream,
    startTranscription: StartRealtimeTranscription,
  ) => {
    const session = recordingSessionRef.current;
    const transcription = startTranscription((text) => {
      if (recordingSessionRef.current !== session) return;
      if (text.trim()) {
        heardSpeechRef.current = true;
        clearTimer(idleTimerRef);
      }
      onInterimTranscript?.(text);
    });
    let capture: PcmCapture;
    try {
      capture = await startPcmCapture(stream, transcription.sendAudio);
    } catch (error) {
      transcription.cancel();
      throw error;
    }
    const realtimeRecording = { capture, transcription };
    realtimeRef.current = realtimeRecording;
    const settle = (text: string) => {
      if (realtimeRef.current !== realtimeRecording) return;
      realtimeRef.current = null;
      capture.stop();
      autoListenRef.current = false;
      cleanupRecording();
      const transcript = text.trim();
      setState("idle");
      if (transcript) onTranscript(transcript);
    };
    void transcription.result.then(settle, (error) => {
      if (realtimeRef.current !== realtimeRecording) return;
      realtimeRef.current = null;
      capture.stop();
      autoListenRef.current = false;
      cleanupRecording();
      setState("idle");
      onError(transcriptionErrorKey(error));
    });
  }, [cleanupRecording, onError, onInterimTranscript, onTranscript]);

  const startRecording = useCallback(async () => {
    if (!onTranscribeAudio || state !== "idle" || startPendingRef.current) return;
    onClearError();
    if (window.isSecureContext === false) {
      onError("insecureContext");
      return;
    }
    const mediaDevices = navigator.mediaDevices;
    const MediaRecorderCtor = mediaRecorderConstructor();
    const startTranscription = realtime ? onStartRealtimeTranscription : undefined;
    if (!mediaDevices?.getUserMedia || (!startTranscription && !MediaRecorderCtor)) {
      onError("unsupported");
      return;
    }
    startPendingRef.current = true;
    try {
      const stream = await mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      if (startTranscription) {
        await startRealtimeRecording(stream, startTranscription);
        startedAtRef.current = Date.now();
        setElapsedMs(0);
        startWaveform(stream);
        setState("recording");
        onClearError();
        maxTimerRef.current = setTimeout(stopRecording, VOICE_RECORDING_MAX_MS);
        if (autoListenRef.current) {
          idleTimerRef.current = setTimeout(() => {
            idleTimerRef.current = null;
            if (!autoListenRef.current || heardSpeechRef.current) return;
            autoListenRef.current = false;
            cleanupRecording();
            setState("idle");
          }, VOICE_AUTO_LISTEN_IDLE_MS);
        }
        return;
      }
      if (!MediaRecorderCtor) throw new DOMException("MediaRecorder is unavailable", "NotSupportedError");
      const recorder = new MediaRecorderCtor(stream, mediaRecorderOptions(MediaRecorderCtor));
      chunksRef.current = [];
      mediaRecorderRef.current = recorder;
      startedAtRef.current = Date.now();
      setElapsedMs(0);
      startWaveform(stream);
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data);
      };
      recorder.onstop = () => {
        const chunks = chunksRef.current.splice(0);
        const durationMs = Math.max(0, Date.now() - startedAtRef.current);
        const mimeType = recorder.mimeType || "audio/webm";
        cleanupRecording();
        if (chunks.length === 0) {
          setState("idle");
          return;
        }
        if (durationMs < VOICE_RECORDING_MIN_MS) {
          setState("idle");
          onError("tooShort");
          return;
        }
        setState("transcribing");
        const blob = new Blob(chunks, { type: mimeType });
        const audioPromise = wantsWav ? convertBlobToWav(blob) : blobToDataUrl(blob);
        void audioPromise
          .then((dataUrl) => onTranscribeAudio(dataUrl, { durationMs, onDelta: onTranscriptDelta }))
          .then(onTranscript)
          .catch((error) => onError(transcriptionErrorKey(error)))
          .finally(() => setState("idle"));
      };
      const session = recordingSessionRef.current;
      const scheduleLiveTranscription = () => {
        liveTimerRef.current = setTimeout(() => {
          liveTimerRef.current = null;
          void transcribeCapturedAudio().finally(() => {
            if (recordingSessionRef.current === session) scheduleLiveTranscription();
          });
        }, VOICE_LIVE_INTERVAL_MS);
      };
      const transcribeCapturedAudio = async () => {
        const durationMs = Date.now() - startedAtRef.current;
        if (chunksRef.current.length === 0 || durationMs < VOICE_RECORDING_MIN_MS) return;
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType || "audio/webm" });
        try {
          const dataUrl = await (wantsWav ? convertBlobToWav(blob) : blobToDataUrl(blob));
          if (recordingSessionRef.current !== session) return;
          const text = await onTranscribeAudio(dataUrl, { durationMs });
          if (recordingSessionRef.current === session) onInterimTranscript?.(text);
        } catch {
          // The final transcription after stop reports any persistent failure.
        }
      };
      if (live) {
        recorder.start(VOICE_LIVE_TIMESLICE_MS);
        scheduleLiveTranscription();
      } else {
        recorder.start();
      }
      setState("recording");
      onClearError();
      maxTimerRef.current = setTimeout(stopRecording, VOICE_RECORDING_MAX_MS);
    } catch (error) {
      autoListenRef.current = false;
      cleanupRecording();
      setState("idle");
      onError(recordingErrorKey(error));
    }
  }, [
    cleanupRecording,
    onClearError,
    onError,
    onTranscribeAudio,
    onTranscript,
    onTranscriptDelta,
    onInterimTranscript,
    live,
    realtime,
    onStartRealtimeTranscription,
    startRealtimeRecording,
    startWaveform,
    state,
    stopRecording,
    wantsWav,
  ]);

  /** Push-to-talk recordings end on release, not on other input. */
  const isHeld = useCallback(() => holdActiveRef.current || shortcutActiveRef.current, []);

  /** Drop the current recording without producing a transcript. */
  const cancelRecording = useCallback(() => {
    if (mediaRecorderRef.current) mediaRecorderRef.current.onstop = null;
    autoListenRef.current = false;
    cleanupRecording();
    setState("idle");
  }, [cleanupRecording]);

  const startAutoListen = useCallback(() => {
    if (state !== "idle") return;
    autoListenRef.current = true;
    heardSpeechRef.current = false;
    void startRecording();
  }, [startRecording, state]);

  const startRecordingWithDeferredStop = useCallback(() => {
    stopAfterStartRef.current = false;
    void startRecording().then(() => {
      if (!stopAfterStartRef.current) return;
      stopAfterStartRef.current = false;
      stopRecording();
    });
  }, [startRecording, stopRecording]);

  const beginPress = useCallback((event: ReactPointerEvent<HTMLButtonElement>) => {
    if (event.pointerType === "mouse" && event.button !== 0) return;
    if (!onTranscribeAudio || disabled || state !== "idle") return;
    clearTimer(holdTimerRef);
    try {
      event.currentTarget.setPointerCapture(event.pointerId);
    } catch {
      // Some embedded runtimes do not expose pointer capture for toolbar buttons.
    }
    holdTimerRef.current = setTimeout(() => {
      holdTimerRef.current = null;
      holdActiveRef.current = true;
      suppressNextClick();
      autoListenRef.current = false;
      startRecordingWithDeferredStop();
    }, VOICE_HOLD_START_MS);
  }, [disabled, onTranscribeAudio, startRecordingWithDeferredStop, state, suppressNextClick]);

  const endPress = useCallback(() => {
    const wasHoldRecording = holdActiveRef.current;
    clearTimer(holdTimerRef);
    if (!wasHoldRecording) return;
    holdActiveRef.current = false;
    suppressNextClick();
    stopRecordingWhenReady();
  }, [stopRecordingWhenReady, suppressNextClick]);

  const handleClick = useCallback(() => {
    if (suppressClickRef.current) {
      clearSuppressClickTimer();
      suppressClickRef.current = false;
      return;
    }
    if (state === "recording") stopRecording();
    else {
      autoListenRef.current = false;
      void startRecording();
    }
  }, [clearSuppressClickTimer, startRecording, state, stopRecording]);

  const beginShortcutHold = useCallback(() => {
    if (!onTranscribeAudio || disabled || state !== "idle" || shortcutActiveRef.current) return;
    shortcutActiveRef.current = true;
    autoListenRef.current = false;
    startRecordingWithDeferredStop();
  }, [disabled, onTranscribeAudio, startRecordingWithDeferredStop, state]);

  const endShortcutHold = useCallback(() => {
    if (!shortcutActiveRef.current) return;
    shortcutActiveRef.current = false;
    stopRecordingWhenReady();
  }, [stopRecordingWhenReady]);

  useEffect(() => {
    if (state !== "recording") {
      setElapsedMs(0);
      return;
    }
    const updateElapsed = () => {
      setElapsedMs(Math.max(0, Date.now() - startedAtRef.current));
    };
    updateElapsed();
    const interval = window.setInterval(updateElapsed, 250);
    return () => window.clearInterval(interval);
  }, [state]);

  useEffect(() => cleanupRecording, [cleanupRecording]);
  useEffect(() => () => clearSuppressClickTimer(), [clearSuppressClickTimer]);

  return {
    beginShortcutHold,
    beginPress,
    cancelRecording,
    buttonDisabled: disabled || state === "transcribing",
    elapsedLabel: formatVoiceElapsed(elapsedMs),
    endShortcutHold,
    endPress,
    handleClick,
    isHeld,
    isRecording: state === "recording",
    levels,
    startAutoListen,
    state,
    stopRecording,
  };
}

interface VoiceAudioState {
  analyser: AnalyserNode;
  context: AudioContext;
  data: Uint8Array<ArrayBuffer>;
  frame: number | null;
  source: MediaStreamAudioSourceNode;
}

function clearTimer(ref: { current: ReturnType<typeof setTimeout> | null }) {
  if (ref.current !== null) {
    clearTimeout(ref.current);
    ref.current = null;
  }
}

function mediaRecorderOptions(MediaRecorderCtor: MediaRecorderConstructor): MediaRecorderOptions | undefined {
  const mimeType = VOICE_MIME_CANDIDATES.find((type) => MediaRecorderCtor.isTypeSupported?.(type));
  return mimeType ? { mimeType } : undefined;
}

type MediaRecorderConstructor = typeof MediaRecorder;

function mediaRecorderConstructor(): MediaRecorderConstructor | undefined {
  if (typeof window === "undefined") return undefined;
  const browserWindow = window as Window & {
    MediaRecorder?: MediaRecorderConstructor;
  };
  return browserWindow.MediaRecorder;
}

function formatVoiceElapsed(ms: number): string {
  const seconds = Math.max(0, Math.floor(ms / 1000));
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${String(seconds % 60).padStart(2, "0")}`;
}

function audioContextConstructor(): typeof AudioContext | undefined {
  if (typeof window === "undefined") return undefined;
  return window.AudioContext
    ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
}

function voiceLevelFromSamples(samples: ArrayLike<number>): number {
  if (samples.length === 0) return 0;
  let sum = 0;
  for (let index = 0; index < samples.length; index += 1) {
    const centered = (samples[index] - 128) / 128;
    sum += centered * centered;
  }
  const rms = Math.sqrt(sum / samples.length);
  return Math.min(1, Math.pow(rms * 4.2, 0.72));
}

function waveformHeightFromLevel(level: number): number {
  if (level < VOICE_MIN_LEVEL) return VOICE_WAVEFORM_SILENT_HEIGHT;
  const activeLevel = Math.min(1, (level - VOICE_MIN_LEVEL) / (1 - VOICE_MIN_LEVEL));
  return Math.round(
    VOICE_WAVEFORM_MIN_HEIGHT
      + activeLevel * (VOICE_WAVEFORM_MAX_HEIGHT - VOICE_WAVEFORM_MIN_HEIGHT),
  );
}

function blobToDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === "string") resolve(reader.result);
      else reject(new Error("invalid_data_url"));
    };
    reader.onerror = () => reject(reader.error ?? new Error("read_failed"));
    reader.readAsDataURL(blob);
  });
}

/**
 * Convert any browser-recorded audio blob (typically webm/opus) to WAV
 * using the Web Audio API. This avoids sending unsupported formats
 * (e.g. webm) to ASR providers that only accept wav/mp3/mpeg.
 */
async function convertBlobToWav(blob: Blob): Promise<string> {
  const AudioCtx = audioContextConstructor();
  if (!AudioCtx) return blobToDataUrl(blob);

  const arrayBuffer = await blob.arrayBuffer();
  const ctx = new AudioCtx({ sampleRate: VOICE_WAV_SAMPLE_RATE });
  try {
    const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
    const wavBlob = audioBufferToWav(audioBuffer);
    return blobToDataUrl(wavBlob);
  } finally {
    void ctx.close();
  }
}

/**
 * Encode an AudioBuffer as a 16-bit PCM WAV Blob.
 */
function audioBufferToWav(buffer: AudioBuffer): Blob {
  const numChannels = buffer.numberOfChannels;
  const sampleRate = buffer.sampleRate;
  const format = 1; // PCM
  const bitsPerSample = 16;

  // Interleave channels
  const channels: Float32Array[] = [];
  for (let ch = 0; ch < numChannels; ch++) {
    channels.push(buffer.getChannelData(ch));
  }
  const length = channels[0].length;
  const interleaved = new Int16Array(length * numChannels);
  for (let i = 0; i < length; i++) {
    for (let ch = 0; ch < numChannels; ch++) {
      const sample = Math.max(-1, Math.min(1, channels[ch][i]));
      interleaved[i * numChannels + ch] = sample < 0
        ? sample * 0x8000
        : sample * 0x7FFF;
    }
  }

  const byteRate = sampleRate * numChannels * (bitsPerSample / 8);
  const blockAlign = numChannels * (bitsPerSample / 8);
  const dataSize = interleaved.byteLength;
  const headerSize = 44;
  const totalSize = headerSize + dataSize;

  const buffer2 = new ArrayBuffer(totalSize);
  const view = new DataView(buffer2);

  // RIFF header
  writeString(view, 0, "RIFF");
  view.setUint32(4, totalSize - 8, true);
  writeString(view, 8, "WAVE");

  // fmt sub-chunk
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true); // sub-chunk size
  view.setUint16(20, format, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);

  // data sub-chunk
  writeString(view, 36, "data");
  view.setUint32(40, dataSize, true);

  new Int16Array(buffer2, headerSize).set(interleaved);

  return new Blob([buffer2], { type: "audio/wav" });
}

function writeString(view: DataView, offset: number, str: string): void {
  for (let i = 0; i < str.length; i++) {
    view.setUint8(offset + i, str.charCodeAt(i));
  }
}

function transcriptionErrorKey(error: unknown): VoiceRecorderErrorKey {
  const detail = error instanceof Error ? error.message : "";
  if (detail === "not_configured") return "notConfigured";
  if (detail === "duration") return "tooLong";
  return "failed";
}

function recordingErrorKey(error: unknown): VoiceRecorderErrorKey {
  const name = error instanceof Error ? error.name : "";
  if (name === "NotFoundError") return "noDevice";
  if (name === "NotSupportedError") return "unsupported";
  return "permission";
}
