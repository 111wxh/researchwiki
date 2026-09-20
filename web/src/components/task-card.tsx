"use client";

import { CircleCheck, Loader2 } from "lucide-react";
import type { TaskData } from "@/lib/chat";

/** 子 agent 任务行（ZCode 风格）：无边框扁平行——图标 + 标题 + 灰色结果。 */
export function TaskCard({ data }: { data: TaskData }) {
  const running = data.status === "running";
  return (
    <div className="fade-in flex items-baseline gap-2 py-0.5 text-[13px]">
      {running ? (
        <Loader2 className="size-3.5 shrink-0 translate-y-0.5 animate-spin text-muted" />
      ) : (
        <CircleCheck className="size-3.5 shrink-0 translate-y-0.5 text-emerald-400/80" />
      )}
      <span className={running ? "text-foreground/90" : "text-foreground/70"}>
        {data.title}
      </span>
      <span className="text-xs text-muted">{data.detail}</span>
    </div>
  );
}
