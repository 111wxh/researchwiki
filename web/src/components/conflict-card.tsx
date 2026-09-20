"use client";

import { TriangleAlert } from "lucide-react";
import type { ConflictData } from "@/lib/chat";

/** 冲突对账行（ZCode 风格）：琥珀色扁平行，绝不静默覆盖。 */
export function ConflictCard({ data }: { data: ConflictData }) {
  return (
    <div className="fade-in flex items-baseline gap-2 py-0.5 text-[13px]">
      <TriangleAlert className="size-3.5 shrink-0 translate-y-0.5 text-amber-400/80" />
      <span className="text-amber-200/80">{data.summary}</span>
      <span className="shrink-0 text-xs text-muted">{data.id}</span>
    </div>
  );
}
