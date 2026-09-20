"use client";

import { useEffect, useRef, useState } from "react";
import { Brain, ChevronRight, Loader2 } from "lucide-react";
import { partText } from "@/lib/chat";

/**
 * 思考过程（ZCode 风格）：无边框扁平行——流式中 spinner + shimmer，
 * 完成后收起为一行"思考 · 持续了 N 秒"，点击展开正文（字色更淡）。
 */
export function Thinking({
  text,
  state,
}: {
  text: string;
  state: "streaming" | "done" | string;
}) {
  const streaming = state === "streaming";
  const [open, setOpen] = useState(streaming);
  const [secs, setSecs] = useState<number | null>(null);
  const startRef = useRef<number | null>(null);

  if (streaming && startRef.current === null) {
    startRef.current = Date.now();
  }

  useEffect(() => {
    if (state === "done" && startRef.current !== null) {
      setSecs(Math.max(1, Math.round((Date.now() - startRef.current) / 1000)));
      setOpen(false);
    }
  }, [state]);

  return (
    <div className="fade-in">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-2 py-1 text-left text-[13px] text-muted transition-colors hover:text-foreground/70"
      >
        {streaming ? (
          <Loader2 className="size-3.5 shrink-0 animate-spin text-muted" />
        ) : (
          <Brain className="size-3.5 shrink-0" />
        )}
        {streaming ? (
          <span className="shimmer-text font-medium">思考中…</span>
        ) : (
          <span>
            思考{secs !== null ? ` · 持续了 ${secs} 秒` : ""}
          </span>
        )}
        <ChevronRight
          className={`size-3.5 shrink-0 transition-transform ${open ? "rotate-90" : ""}`}
        />
      </button>
      {open && (
        <div className="pl-6 pr-2 pb-1.5 text-[13px] leading-relaxed text-muted/60 whitespace-pre-wrap">
          {text}
          {streaming && <span className="cursor-blink">▍</span>}
        </div>
      )}
    </div>
  );
}
