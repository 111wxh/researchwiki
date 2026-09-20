"use client";

import { useEffect, useRef } from "react";
import Markdown from "react-markdown";
import {
  BrainCircuit,
  Database,
  FileSearch,
  ListChecks,
  Loader2,
} from "lucide-react";
import { Thinking } from "@/components/thinking";
import { TaskCard } from "@/components/task-card";
import { NoteCard } from "@/components/note-card";
import { ConflictCard } from "@/components/conflict-card";
import { ProcessBlock } from "@/components/process-block";
import { ReportView, type SourceItem } from "@/components/report-view";
import { WikiPanel } from "@/components/wiki-panel";
import {
  partText,
  useResearchChat,
  type ConflictData,
  type NoteData,
  type TaskData,
} from "@/lib/chat";

const EXAMPLES = [
  "2026 年 agent 记忆方案对比",
  "上下文压缩的主流做法",
  "SQLite 向量检索方案",
];

const FEATURES = [
  { icon: ListChecks, label: "过程可视", desc: "思考 · 任务 · 时间线" },
  { icon: FileSearch, label: "引用可溯", desc: "断言溯源到笔记与来源" },
  { icon: Database, label: "知识沉淀", desc: "结论写入个人 wiki" },
];

function asData<T>(part: unknown): T | null {
  return (part as { data?: T })?.data ?? null;
}

export default function Home() {
  const { messages, status, sendMessage, stop } = useResearchChat();
  const isLoading = status === "streaming" || status === "submitted";
  const scrollRef = useRef<HTMLDivElement>(null);
  const stickToBottomRef = useRef(true);

  // 智能滚动：仅当用户本就在底部时跟随流式输出，不抢滚动条
  const onScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    stickToBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
  };

  useEffect(() => {
    const el = scrollRef.current;
    if (el && stickToBottomRef.current) {
      el.scrollTo({ top: el.scrollHeight });
    }
  }, [messages, status]);

  // 聚合全会话的笔记与冲突 → Wiki 面板
  const notes: NoteData[] = [];
  const conflicts: ConflictData[] = [];
  for (const m of messages) {
    if (m.role !== "assistant") continue;
    for (const part of m.parts as unknown[]) {
      const t = (part as { type: string }).type;
      if (t === "data-note") {
        const d = asData<NoteData>(part);
        if (d) notes.push(d);
      } else if (t === "data-conflict") {
        const d = asData<ConflictData>(part);
        if (d) conflicts.push(d);
      }
    }
  }

  return (
    <div className="flex h-dvh flex-col">
      {/* 顶栏 */}
      <header className="flex items-center gap-3 border-b border-border-subtle bg-surface/60 px-5 py-3 backdrop-blur">
        <div className="flex size-8 items-center justify-center rounded-lg bg-accent/15 text-accent">
          <BrainCircuit className="size-[18px]" />
        </div>
        <div>
          <div className="text-sm font-semibold">ResearchWiki</div>
          <div className="text-[11px] text-muted">自进化研究 Wiki 智能体</div>
        </div>
        <span className="ml-auto rounded-full border border-border-subtle bg-surface-2 px-2.5 py-1 text-[11px] text-muted">
          Mock 模式 · 阶段 1 骨架
        </span>
      </header>

      <div className="flex min-h-0 flex-1">
        {/* 消息流 */}
        <main
          ref={scrollRef}
          onScroll={onScroll}
          className="flex-1 overflow-y-auto px-4 py-6"
        >
          <div className="mx-auto max-w-3xl space-y-4">
            {messages.length === 0 && (
              <div className="pt-[16vh] text-center">
                <div className="mx-auto flex size-12 items-center justify-center rounded-2xl bg-accent/15 text-accent">
                  <BrainCircuit className="size-6" />
                </div>
                <h1 className="mt-4 text-xl font-semibold">给它一个研究问题</h1>
                <p className="mx-auto mt-2 max-w-md text-sm leading-relaxed text-muted">
                  它将长程地搜索、阅读、综合，把结论沉淀成带交叉引用的个人 wiki——
                  越用越快、越用越准。
                </p>
                <div className="mx-auto mt-6 flex max-w-lg justify-center gap-3">
                  {FEATURES.map((f) => (
                    <div
                      key={f.label}
                      className="flex-1 rounded-xl border border-border-subtle bg-surface/60 px-3 py-3"
                    >
                      <f.icon className="mx-auto size-4 text-accent" />
                      <div className="mt-1.5 text-[13px] font-medium">{f.label}</div>
                      <div className="mt-0.5 text-[11px] leading-snug text-muted">
                        {f.desc}
                      </div>
                    </div>
                  ))}
                </div>
                <div className="mt-6 flex flex-wrap justify-center gap-2">
                  {EXAMPLES.map((q) => (
                    <button
                      key={q}
                      type="button"
                      onClick={() => sendMessage({ text: q })}
                      className="rounded-full border border-border-subtle bg-surface px-3.5 py-1.5 text-[13px] text-muted transition-colors hover:border-accent/50 hover:text-foreground"
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((m) => {
              if (m.role === "user") {
                return (
                  <div key={m.id} className="flex justify-end">
                    <div className="fade-in max-w-[80%] rounded-2xl rounded-br-md bg-surface-2 px-4 py-2.5 text-sm">
                      {m.parts.map((p, i) =>
                        p.type === "text" ? <span key={i}>{p.text}</span> : null,
                      )}
                    </div>
                  </div>
                );
              }

              // source-url parts → 引用悬浮卡与底部来源列表的数据
              const sources: SourceItem[] = (m.parts as unknown[])
                .filter((p) => (p as { type: string }).type === "source-url")
                .map((p, i) => {
                  const s = p as { sourceId?: string; url: string; title?: string };
                  return { id: s.sourceId ?? String(i), url: s.url, title: s.title ?? s.url };
                });

              return (
                <div key={m.id} className="space-y-1.5">
                  {(() => {
                    // 分桶：思考行 / 过程操作行 / 正文，渲染顺序与流内一致
                    const reasonings: unknown[] = [];
                    const ops: React.ReactNode[] = [];
                    const texts: React.ReactNode[] = [];
                    let tasks = 0;
                    let noteCount = 0;
                    let conflictCount = 0;

                    for (const [i, part] of (m.parts as unknown[]).entries()) {
                      const type = (part as { type: string }).type;
                      if (type === "reasoning") {
                        reasonings.push(
                          <Thinking
                            key={i}
                            text={partText(part)}
                            state={(part as { state?: string }).state ?? "done"}
                          />,
                        );
                      } else if (type === "data-task") {
                        const d = asData<TaskData>(part);
                        if (d) {
                          tasks += 1;
                          ops.push(<TaskCard key={i} data={d} />);
                        }
                      } else if (type === "data-note") {
                        const d = asData<NoteData>(part);
                        if (d) {
                          noteCount += 1;
                          ops.push(<NoteCard key={i} data={d} compact />);
                        }
                      } else if (type === "data-conflict") {
                        const d = asData<ConflictData>(part);
                        if (d) {
                          conflictCount += 1;
                          ops.push(<ConflictCard key={i} data={d} />);
                        }
                      } else if (type === "text") {
                        const text = (part as { text: string }).text ?? "";
                        const streaming =
                          (part as { state?: string }).state === "streaming";
                        if (text.startsWith("## ")) {
                          texts.push(
                            <ReportView
                              key={i}
                              text={text}
                              streaming={streaming}
                              sources={sources}
                              notes={notes}
                            />,
                          );
                        } else {
                          texts.push(
                            <div key={i} className="md-report pt-1 text-sm">
                              <Markdown>{text}</Markdown>
                            </div>,
                          );
                        }
                      }
                    }

                    const isActive =
                      isLoading && m.id === messages[messages.length - 1]?.id;

                    return (
                      <>
                        {reasonings}
                        {ops.length > 0 && (
                          <ProcessBlock
                            active={isActive}
                            summary={{
                              tasks,
                              notes: noteCount,
                              conflicts: conflictCount,
                            }}
                          >
                            {ops}
                          </ProcessBlock>
                        )}
                        {texts}
                      </>
                    );
                  })()}
                </div>
              );
            })}

            {/* 已提交、尚未收到首个事件 */}
            {status === "submitted" &&
              messages.length > 0 &&
              messages[messages.length - 1].role === "user" && (
                <div className="fade-in flex items-center gap-2 px-1 text-[13px] text-muted">
                  <Loader2 className="size-3.5 animate-spin text-accent" />
                  正在建立研究会话…
                </div>
              )}
          </div>
        </main>

        <WikiPanel notes={notes} conflicts={conflicts} growing={isLoading} />
      </div>

      {/* 输入区 */}
      <footer className="border-t border-border-subtle bg-surface/60 px-4 py-3.5 backdrop-blur">
        <form
          className="mx-auto flex max-w-3xl items-end gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            const textarea = e.currentTarget.elements.namedItem("q") as HTMLTextAreaElement;
            const value = textarea.value.trim();
            if (!value || isLoading) return;
            stickToBottomRef.current = true;
            sendMessage({ text: value });
            textarea.value = "";
          }}
        >
          <textarea
            name="q"
            rows={1}
            placeholder="输入研究问题，Enter 发送，Shift+Enter 换行"
            className="max-h-32 flex-1 resize-none rounded-xl border border-border-subtle bg-surface px-4 py-2.5 text-sm outline-none placeholder:text-muted/60 focus:border-accent/60"
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                e.currentTarget.form?.requestSubmit();
              }
            }}
          />
          {isLoading ? (
            <button
              type="button"
              onClick={() => stop()}
              className="rounded-xl border border-border-subtle bg-surface-2 px-4 py-2.5 text-sm text-muted hover:text-foreground"
            >
              停止
            </button>
          ) : (
            <button
              type="submit"
              className="rounded-xl bg-accent px-4 py-2.5 text-sm font-medium text-white transition-opacity hover:opacity-90"
            >
              研究
            </button>
          )}
        </form>
      </footer>
    </div>
  );
}
