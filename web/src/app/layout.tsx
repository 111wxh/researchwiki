import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ResearchWiki · 自进化研究 Wiki 智能体",
  description:
    "给它一个研究问题，它长程地搜索、阅读、综合，把结论沉淀成带交叉引用的个人 wiki——越用越快越准。",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="zh-CN" className="h-full antialiased">
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
