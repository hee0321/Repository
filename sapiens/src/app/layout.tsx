import type { Metadata } from "next";
import {
  Noto_Sans_KR,
  Noto_Serif_KR,
  IBM_Plex_Mono,
} from "next/font/google";
import "./globals.css";

const notoSans = Noto_Sans_KR({
  variable: "--font-noto-sans",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  display: "swap",
});

const notoSerif = Noto_Serif_KR({
  variable: "--font-noto-serif",
  subsets: ["latin"],
  weight: ["400", "600", "700"],
  display: "swap",
});

const ibmMono = IBM_Plex_Mono({
  variable: "--font-ibm-mono",
  subsets: ["latin"],
  weight: ["400", "500"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "사피엔스 · 사고의 실험실",
  description:
    "노트, 결정, 봇이 한 책상 위에 모이는 사고의 실험실. 현희의 작은 생각 워크벤치.",
  openGraph: {
    title: "사피엔스 · 사고의 실험실",
    description:
      "노트, 결정, 봇이 한 책상 위에 모이는 사고의 실험실.",
    locale: "ko_KR",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="ko"
      className={`${notoSans.variable} ${notoSerif.variable} ${ibmMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col bg-background text-foreground">
        {children}
      </body>
    </html>
  );
}
