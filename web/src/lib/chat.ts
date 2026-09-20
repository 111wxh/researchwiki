"use client";

import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";

/** 后端自定义 data parts 的载荷类型（对应 loop/research_run.py 发出的事件） */

export interface TaskData {
  title: string;
  status: "running" | "done";
  detail: string;
}

export interface NoteData {
  id: string;
  text: string;
  entities: string[];
  confidence: "high" | "medium" | "low";
}

export interface ConflictData {
  id: string;
  summary: string;
  action: string;
}

export const BACKEND_URL = "http://localhost:8000/api/chat";

export function useResearchChat() {
  return useChat({
    transport: new DefaultChatTransport({ api: BACKEND_URL }),
  });
}

/** 消息 parts 的防御式取值：reasoning 字段随 SDK 版本可能叫 reasoning 或 text */
export function partText(part: unknown): string {
  const p = part as { reasoning?: string; text?: string };
  return p?.reasoning ?? p?.text ?? "";
}
