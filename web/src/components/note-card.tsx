"use client";

import type { ConflictData, NoteData } from "@/lib/chat";

const confidenceColor: Record<NoteData["confidence"], string> = {
  high: "bg-emerald-500",
  medium: "bg-amber-500",
  low: "bg-zinc-500",
};

/** 原子笔记卡：wiki 在生长的可视化单元。compact 为流内扁平行（ZCode 风格）。 */
export function NoteCard({ data, compact = false }: { data: NoteData; compact?: boolean }) {
  if (compact) {
    return (
      <div className="fade-in flex items-baseline gap-2 py-0.5 text-[13px]">
        <span className="shrink-0 font-mono text-[11px] text-accent/80">{data.id}</span>
        <span className="line-clamp-2 text-muted">{data.text}</span>
      </div>
    );
  }
  return (
    <div className="fade-in rounded-lg border border-border-subtle bg-surface px-3.5 py-3">
      <div className="flex items-center gap-2">
        <span className="rounded bg-surface-2 px-1.5 py-0.5 font-mono text-[11px] text-accent">
          {data.id}
        </span>
        <span className="flex items-center gap-1 text-[10px] text-muted">
          <span className={`size-1.5 rounded-full ${confidenceColor[data.confidence]}`} />
          {data.confidence}
        </span>
      </div>
      <p className="mt-1.5 text-[13px] leading-relaxed text-muted">{data.text}</p>
      {data.entities.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {data.entities.map((e) => (
            <span
              key={e}
              className="rounded-full border border-border-subtle px-2 py-0.5 text-[11px] text-muted"
            >
              {e}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
