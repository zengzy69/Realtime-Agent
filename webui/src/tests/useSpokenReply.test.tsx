import { act, renderHook } from "@testing-library/react";
import type { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { useSpokenReply } from "@/hooks/useSpokenReply";
import type { SpeechHandlers } from "@/lib/types";
import { ClientProvider } from "@/providers/ClientProvider";

const players = vi.hoisted(() => [] as Array<{
  sampleRate: number;
  push: ReturnType<typeof vi.fn>;
  drain: ReturnType<typeof vi.fn>;
  stop: ReturnType<typeof vi.fn>;
}>);

vi.mock("@/lib/pcm-player", () => ({
  createPcmPlayer: (sampleRate: number) => {
    const player = {
      sampleRate,
      push: vi.fn(),
      drain: vi.fn(() => Promise.resolve()),
      stop: vi.fn(),
    };
    players.push(player);
    return player;
  },
}));

function fakeClient() {
  const streams: Array<{ text: string; handlers: SpeechHandlers; stop: ReturnType<typeof vi.fn> }> = [];
  const client = {
    speak: (text: string, handlers: SpeechHandlers) => {
      const stream = { text, handlers, stop: vi.fn() };
      streams.push(stream);
      return { stop: stream.stop };
    },
  };
  const wrapper = ({ children }: { children: ReactNode }) => (
    <ClientProvider
      client={client as unknown as import("@/lib/nanobot-client").NanobotClient}
      token="tok"
    >
      {children}
    </ClientProvider>
  );
  return { streams, wrapper };
}

describe("useSpokenReply", () => {
  beforeEach(() => {
    players.length = 0;
  });

  it("plays the excerpt and returns to idle after the audio drains", async () => {
    const { streams, wrapper } = fakeClient();
    const { result } = renderHook(() => useSpokenReply(), { wrapper });

    act(() => result.current.speak("Reply.\n\nDetails."));
    expect(result.current.state).toEqual({ status: "preparing" });
    const [{ handlers }] = streams;
    act(() => handlers.onStart({ text: "Reply.", truncated: true, sampleRate: 24000 }));
    expect(result.current.state).toEqual({
      status: "speaking",
      text: "Reply.",
      truncated: true,
    });
    const pcm = new Int16Array([1, 2]);
    act(() => handlers.onAudio(pcm));
    await act(async () => handlers.onEnd());

    expect(players[0].sampleRate).toBe(24000);
    expect(players[0].push).toHaveBeenCalledWith(pcm);
    expect(players[0].stop).toHaveBeenCalled();
    expect(result.current.state).toEqual({ status: "idle" });
  });

  it("reports completed, empty, error, and replaced settlements", async () => {
    const { streams, wrapper } = fakeClient();
    const { result } = renderHook(() => useSpokenReply(), { wrapper });
    const onSettled = vi.fn();

    act(() => result.current.speak("First.", { onSettled }));
    act(() => streams[0].handlers.onStart({ text: "First.", truncated: false, sampleRate: 24000 }));
    act(() => result.current.speak("Second.", { onSettled }));
    expect(onSettled).toHaveBeenCalledWith("stopped");

    act(() => streams[1].handlers.onError("empty"));
    expect(onSettled).toHaveBeenLastCalledWith("empty");

    act(() => result.current.speak("Third.", { onSettled }));
    act(() => streams[2].handlers.onError("provider_error"));
    expect(onSettled).toHaveBeenLastCalledWith("error");

    act(() => result.current.speak("Fourth.", { onSettled }));
    act(() => streams[3].handlers.onStart({ text: "Fourth.", truncated: false, sampleRate: 24000 }));
    await act(async () => streams[3].handlers.onEnd());
    expect(onSettled).toHaveBeenLastCalledWith("completed");
  });

  it("replaces the current reply and ignores the stale stream", () => {
    const { streams, wrapper } = fakeClient();
    const { result } = renderHook(() => useSpokenReply(), { wrapper });

    act(() => result.current.speak("First."));
    act(() => streams[0].handlers.onStart({ text: "First.", truncated: false, sampleRate: 24000 }));
    act(() => result.current.speak("Second."));
    act(() => streams[0].handlers.onAudio(new Int16Array([1])));

    expect(streams[0].stop).toHaveBeenCalled();
    expect(players[0].stop).toHaveBeenCalled();
    expect(players[0].push).not.toHaveBeenCalled();
    expect(result.current.state).toEqual({ status: "preparing" });
  });

  it("surfaces synthesis failures but stays quiet when nothing is speakable", () => {
    const { streams, wrapper } = fakeClient();
    const { result } = renderHook(() => useSpokenReply(), { wrapper });

    act(() => result.current.speak("```code```"));
    act(() => streams[0].handlers.onError("empty"));
    expect(result.current.state).toEqual({ status: "idle" });

    act(() => result.current.speak("Reply."));
    act(() => streams[1].handlers.onError("provider_error"));
    expect(result.current.state).toEqual({ status: "error", detail: "provider_error" });
  });
});
