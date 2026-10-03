import { AlertTriangle, Loader2, Volume2, X } from "lucide-react";
import { useTranslation } from "react-i18next";

import { Button } from "@/components/ui/button";
import type { SpokenReplyState } from "@/hooks/useSpokenReply";
import { cn } from "@/lib/utils";

interface SpokenReplyNoticeProps {
  state: Exclude<SpokenReplyState, { status: "idle" }>;
  onStop: () => void;
}

/** Status strip above the composer while a reply's opening is read aloud. */
export function SpokenReplyNotice({ state, onStop }: SpokenReplyNoticeProps) {
  const { t } = useTranslation();
  const failed = state.status === "error";
  const Icon = failed ? AlertTriangle : state.status === "preparing" ? Loader2 : Volume2;

  return (
    <div
      role="status"
      aria-live="polite"
      className={cn(
        "mb-2 flex items-center gap-2 rounded-lg border px-3 py-1.5 text-[12px] leading-5",
        "animate-in fade-in-0 slide-in-from-bottom-1",
        failed
          ? "border-destructive/30 bg-destructive/10 text-destructive"
          : "border-border bg-muted/60 text-muted-foreground",
      )}
    >
      <Icon
        className={cn("h-4 w-4 shrink-0", state.status === "preparing" && "animate-spin")}
        aria-hidden
      />
      <p className="flex-1">
        {failed
          ? t("thread.spokenReply.failed")
          : state.status === "preparing"
            ? t("thread.spokenReply.preparing")
            : state.truncated
              ? t("thread.spokenReply.speakingSummary")
              : t("thread.spokenReply.speaking")}
      </p>
      <Button
        variant="ghost"
        size="icon"
        onClick={onStop}
        aria-label={failed ? t("common.dismiss") : t("thread.spokenReply.stop")}
        className="h-6 w-6 shrink-0"
      >
        <X className="h-3.5 w-3.5" />
      </Button>
    </div>
  );
}
