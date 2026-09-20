"use client";

import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { FileText, Link2 } from "lucide-react";
import type { NoteData } from "@/lib/chat";

export interface SourceItem {
  id: string;
  url: string;
  title: string;
}

const confidenceColor: Record<NoteData["confidence"], string> = {
  high: "bg-emerald-500",
  medium: "bg-amber-500",
  low: "bg-zinc-500",
};

/** 引用上标 chip：hover 弹出对应来源 + 原子笔记卡。 */
function CitationChip({
  index,
  source,
  note,
}: {
  index: number;
  source?: SourceItem;
  note?: NoteData;
}) {
  return (
    <span className="group relative inline-block align-super">
      <span className="mx-0.5 inline-flex h-4 min-w-4 cursor-default items-center justify-center rounded border border-border-subtle bg-surface-2 px-1 font-mono text-[10px] leading-none text-accent transition-colors group-hover:border-accent/50">
        {index}
      </span>
      <span className="pointer-events-none absolute bottom-full left-1/2 z-10 mb-2 hidden w-72 -translate-x-1/2 rounded-xl border border-border-subtle bg-surface-2 p-3 text-left shadow-2xl shadow-black/50 group-hover:block">
        {note && (
          <span className="block">
            <span className="flex items-center gap-2">
              <span className="rounded bg-surface px-1.5 py-0.5 font-mono text-[10px] text-accent">
                {note.id}
              </span>
              <span className="flex items-center gap-1 text-[10px] text-muted">
                <span className={`size-1.5 rounded-full ${confidenceColor[note.confidence]}`} />
                {note.confidence}
              </span>
            </span>
            <span className="mt-1.5 block text-xs leading-relaxed text-foreground/90">
              {note.text}
            </span>
          </span>
        )}
        {source && (
          <span className="mt-2 flex items-start gap-1.5 border-t border-border-subtle pt-2 text-[11px]">
            <Link2 className="mt-0.5 size-3 shrink-0 text-muted" />
            <span className="block">
              <span className="block text-foreground/90">{source.title}</span>
              <span className="block truncate text-muted/70">{source.url}</span>
            </span>
          </span>
        )}
      </span>
    </span>
  );
}

/** 把正文里的 [n] 引用编号替换为可渲染的内部链接标记。 */
function citeize(text: string): string {
  return text.replace(/\[(\d+)\]/g, (_m, d) => `[${d}](#cite-${d})`);
}

/** 报告文档视图：正文 Markdown + 行内引用悬浮卡 + 底部来源列表。 */
export function ReportView({
  text,
  streaming,
  sources,
  notes,
}: {
  text: string;
  streaming: boolean;
  sources: SourceItem[];
  notes: NoteData[];
}) {
  return (
    <div className="fade-in overflow-hidden rounded-xl border border-border-subtle bg-surface">
      <div className="flex items-center gap-2 border-b border-border-subtle bg-surface-2/60 px-4 py-2.5 text-xs text-muted">
        <FileText className="size-3.5 shrink-0" />
        <span className="font-medium text-foreground">研究报告</span>
        {streaming ? (
          <>
            <span className="shimmer-text">撰写中…</span>
            <span className="ml-auto h-0.5 w-24 rounded-full shimmer-bar" />
          </>
        ) : (
          <span className="ml-auto">{sources.length} 个来源</span>
        )}
      </div>
      <div className="px-5 py-4">
        <div className="md-report">
          <Markdown
            remarkPlugins={[remarkGfm]}
            components={{
              a: ({ href, children }) => {
                const m = /^#cite-(\d+)$/.exec(href ?? "");
                if (!m) {
                  return (
                    <a href={href} target="_blank" rel="noopener noreferrer">
                      {children}
                    </a>
                  );
                }
                const idx = Number(m[1]);
                return (
                  <CitationChip
                    index={idx}
                    source={sources[idx - 1]}
                    note={notes[idx - 1]}
                  />
                );
              },
            }}
          >
            {citeize(text)}
          </Markdown>
          {streaming && <span className="cursor-blink text-accent">▍</span>}
        </div>
        {sources.length > 0 && !streaming && (
          <div className="mt-4 border-t border-border-subtle pt-3">
            <div className="mb-2 text-xs font-medium text-muted">来源</div>
            <ol className="space-y-1.5">
              {sources.map((s, i) => (
                <li key={s.id} className="flex items-baseline gap-2 text-[13px]">
                  <span className="font-mono text-xs text-muted">[{i + 1}]</span>
                  <a
                    href={s.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-accent hover:underline"
                  >
                    {s.title}
                  </a>
                  <span className="truncate text-xs text-muted/70">{s.url}</span>
                </li>
              ))}
            </ol>
          </div>
        )}
      </div>
    </div>
  );
}
