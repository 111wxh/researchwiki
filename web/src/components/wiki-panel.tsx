"use client";

import { Database, StickyNote, TriangleAlert } from "lucide-react";
import type { ConflictData, NoteData } from "@/lib/chat";
import { NoteCard } from "./note-card";

/**
 * Wiki 沉淀面板：聚合整个会话产生的原子笔记与冲突，
 * 研究进行中实时生长——这是"自进化"的直接可视化。
 */
export function WikiPanel({
  notes,
  conflicts,
  growing,
}: {
  notes: NoteData[];
  conflicts: ConflictData[];
  growing: boolean;
}) {
  return (
    <aside className="hidden w-[320px] shrink-0 flex-col border-l border-border-subtle bg-surface/50 lg:flex">
      <div className="border-b border-border-subtle px-4 py-3">
        <div className="flex items-center gap-2">
          <Database className={`size-4 ${growing ? "text-accent" : "text-muted"}`} />
          <span className="text-sm font-semibold">Wiki 知识库</span>
          {growing && <span className="shimmer-text text-xs">生长中…</span>}
        </div>
        <p className="mt-1 text-[11px] leading-relaxed text-muted">
          原子笔记随研究实时写入；页面/来源浏览在阶段 2 接入真实存储后开放。
        </p>
      </div>

      <div className="flex gap-2 px-4 py-2.5 text-[11px] text-muted">
        <span className="flex items-center gap-1.5 rounded-full bg-surface-2 px-2.5 py-1">
          <StickyNote className="size-3" />
          笔记 <span className="font-mono text-foreground">{notes.length}</span>
        </span>
        <span className="flex items-center gap-1.5 rounded-full bg-surface-2 px-2.5 py-1">
          <TriangleAlert className="size-3" />
          冲突 <span className="font-mono text-foreground">{conflicts.length}</span>
        </span>
      </div>

      <div className="flex-1 space-y-2.5 overflow-y-auto px-4 pb-4">
        {notes.length === 0 && (
          <p className="pt-8 text-center text-xs text-muted/60">
            发起一次研究，观察笔记如何沉淀
          </p>
        )}
        {notes.map((n) => (
          <NoteCard key={n.id} data={n} />
        ))}
        {conflicts.map((c) => (
          <div
            key={c.id}
            className="fade-in rounded-lg border border-amber-500/30 bg-amber-500/[0.06] px-3 py-2 text-xs"
          >
            <span className="flex items-center gap-1.5 font-medium text-amber-400">
              <TriangleAlert className="size-3" />
              {c.id}
            </span>
            <p className="mt-1 leading-relaxed text-muted">{c.summary}</p>
          </div>
        ))}
      </div>
    </aside>
  );
}
