"use client";

import { useEffect, useState } from "react";
import { ChevronRight, Loader2, Workflow } from "lucide-react";

/**
 * 研究过程折叠块：任务 / 笔记 / 冲突行收进一个可折叠容器。
 * 收起态对齐 ZCode 的"7 个文件已更新 +220 -46"摘要行——
 * 浅底 pill + 计数；流式进行中展开实时可见，结束后自动收起。
 */
export function ProcessBlock({
  active,
  summary,
  children,
}: {
  active: boolean;
  summary: { tasks: number; notes: number; conflicts: number };
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(active);

  useEffect(() => {
    if (!active) setOpen(false);
  }, [active]);

  const { tasks, notes, conflicts } = summary;
  const counts = [
    tasks > 0 && `${tasks} 个任务`,
    notes > 0 && `${notes} 条笔记`,
    conflicts > 0 && `${conflicts} 处冲突`,
  ]
    .filter(Boolean)
    .join(" · ");

  return (
    <div className="fade-in">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-fit items-center gap-2 rounded-md bg-surface-2/70 px-2.5 py-1.5 text-left text-[13px] text-muted transition-colors hover:bg-surface-2 hover:text-foreground/70"
      >
        {active ? (
          <Loader2 className="size-3.5 shrink-0 animate-spin" />
        ) : (
          <Workflow className="size-3.5 shrink-0" />
        )}
        {active ? (
          <span className="shimmer-text font-medium">研究过程中…</span>
        ) : (
          <span>研究过程 · {counts}</span>
        )}
        <ChevronRight
          className={`size-3.5 shrink-0 transition-transform ${open ? "rotate-90" : ""}`}
        />
      </button>
      {open && <div className="space-y-0.5 py-1.5 pl-1">{children}</div>}
    </div>
  );
}
