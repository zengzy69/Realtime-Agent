import { useCallback, useEffect, useRef, useState } from "react";

import { createPcmPlayer, type PcmPlayer } from "@/lib/pcm-player";
import type { SpeechSynthesis } from "@/lib/types";
import { useClient } from "@/providers/ClientProvider";

export type SpokenReplyState =
  | { status: "idle" }
  | { status: "preparing" }
  | { status: "speaking"; text: string; truncated: boolean }
  | { status: "error"; detail: string };

export type SpokenReplySettlement = "completed" | "stopped" | "error" | "empty";

interface SpokenReplySession {
  synthesis: SpeechSynthesis | null;
  player: PcmPlayer | null;
  settled: boolean;
  onSettled?: (reason: SpokenReplySettlement) => void;
}

/** Plays the gateway's spoken excerpt of a reply; a new `speak` replaces the current one. */
export function useSpokenReply(): {
  state: SpokenReplyState;
  speak: (reply: string, options?: { onSettled?: (reason: SpokenReplySettlement) => void }) => void;
  stop: () => void;
} {
  const { client } = useClient();
  const [state, setState] = useState<SpokenReplyState>({ status: "idle" });
  const activeRef = useRef<SpokenReplySession | null>(null);

  const settle = useCallback((session: SpokenReplySession, reason: SpokenReplySettlement) => {
    if (session.settled) return;
    session.settled = true;
    session.onSettled?.(reason);
  }, []);

  const release = useCallback((reason: SpokenReplySettlement = "stopped") => {
    const active = activeRef.current;
    activeRef.current = null;
    if (active) settle(active, reason);
    active?.synthesis?.stop();
    active?.player?.stop();
  }, [settle]);

  const stop = useCallback(() => {
    release("stopped");
    setState((previous) => (previous.status === "idle" ? previous : { status: "idle" }));
  }, [release]);

  const speak = useCallback((
    reply: string,
    options?: { onSettled?: (reason: SpokenReplySettlement) => void },
  ) => {
    release("stopped");
    const session: SpokenReplySession = {
      synthesis: null,
      player: null,
      settled: false,
      onSettled: options?.onSettled,
    };
    activeRef.current = session;
    const current = () => activeRef.current === session;
    const finish = (next: SpokenReplyState, reason: SpokenReplySettlement) => {
      session.player?.stop();
      activeRef.current = null;
      settle(session, reason);
      setState(next);
    };
    setState({ status: "preparing" });
    session.synthesis = client.speak(reply, {
      onStart: (start) => {
        if (!current()) return;
        session.player = createPcmPlayer(start.sampleRate);
        setState({ status: "speaking", text: start.text, truncated: start.truncated });
      },
      onAudio: (pcm) => {
        if (current()) session.player?.push(pcm);
      },
      onEnd: () => {
        if (!current()) return;
        void (session.player?.drain() ?? Promise.resolve()).then(() => {
          if (current()) finish({ status: "idle" }, "completed");
        });
      },
      onError: (detail) => {
        if (!current()) return;
        if (detail === "empty") finish({ status: "idle" }, "empty");
        else finish({ status: "error", detail }, "error");
      },
    });
  }, [client, release, settle]);

  useEffect(() => () => release("stopped"), [release]);

  return { state, speak, stop };
}
