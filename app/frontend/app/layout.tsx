import type { Metadata } from "next";
import { QueryProvider } from "@/components/QueryProvider";
import "./globals.css";

export const metadata: Metadata = {
  title: "Viral Clipper — AI 9:16 Shorts Generator",
  description: "Extract viral moments from 1h+ videos, reframe to 9:16 portrait, and burn animated captions.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen antialiased bg-[#09090b] text-zinc-100 selection:bg-indigo-500/30 selection:text-indigo-200">
        <QueryProvider>
          <div className="relative min-h-screen flex flex-col justify-between">
            {/* Ambient Background Glow */}
            <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[350px] bg-gradient-to-b from-indigo-600/20 via-purple-600/10 to-transparent blur-[120px] pointer-events-none -z-10" />
            
            <header className="border-b border-zinc-800/60 backdrop-blur-sm sticky top-0 z-50 bg-[#09090b]/80">
              <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
                <a href="/" className="flex items-center gap-2 font-bold text-xl tracking-tight">
                  <span className="w-8 h-8 rounded-lg bg-gradient-to-tr from-indigo-500 to-purple-500 flex items-center justify-center text-white shadow-lg shadow-indigo-500/25">⚡</span>
                  <span>Viral<span className="text-indigo-400">Clipper</span></span>
                </a>
                <div className="flex items-center gap-2">
                  <div className="hidden sm:block text-xs text-zinc-400 font-medium border border-zinc-800 px-3 py-1.5 rounded-full bg-zinc-900/50">
                    9:16 Shorts & Reels AI
                  </div>
                </div>
              </div>
            </header>

            <main className="flex-1 max-w-6xl w-full mx-auto px-6 py-10">
              {children}
            </main>

            <footer className="border-t border-zinc-900 py-6 text-xs text-zinc-600 flex flex-col items-center gap-3">
              <span>Viral Clipper — AI 9:16 Shorts Generator</span>
            </footer>
          </div>
        </QueryProvider>
      </body>
    </html>
  );
}
