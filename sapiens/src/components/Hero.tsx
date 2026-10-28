export function Hero() {
  return (
    <section
      id="top"
      className="grain desk-grid relative overflow-hidden border-b border-border"
    >
      <div className="pointer-events-none absolute -right-24 -top-24 h-72 w-72 rounded-full bg-accent-soft blur-3xl" />
      <div className="pointer-events-none absolute -bottom-20 -left-16 h-64 w-64 rounded-full bg-[#f0e0c8] blur-3xl" />

      <div className="relative z-10 mx-auto grid max-w-5xl gap-10 px-5 py-20 sm:px-8 sm:py-28 lg:grid-cols-[1.2fr_0.8fr] lg:items-end">
        <div>
          <p className="section-label mb-5">Thought Laboratory · Seoul</p>
          <h1 className="font-serif text-4xl font-semibold leading-[1.15] tracking-tight text-ink sm:text-5xl lg:text-[3.4rem]">
            노트, 결정, 봇이
            <br />
            <span className="text-accent">한 책상</span> 위에 모이는
            <br className="hidden sm:block" />
            사고의 실험실
          </h1>
          <p className="mt-6 max-w-xl text-base leading-relaxed text-muted sm:text-lg">
            사피엔스는 현희의 작은 생각 워크벤치입니다. 읽고 고르고 자동화하는
            과정을 같은 책상에 올려 두고, 솔직한 노트만 남깁니다.
          </p>
          <div className="mt-9 flex flex-wrap items-center gap-3">
            <a
              href="#projects"
              className="inline-flex h-11 items-center justify-center rounded-full bg-accent px-6 text-sm font-medium text-white transition hover:bg-[#264a3c]"
            >
              실험대 보기
            </a>
            <a
              href="#contact"
              className="inline-flex h-11 items-center justify-center rounded-full border border-border bg-surface px-6 text-sm font-medium text-ink transition hover:bg-surface-2"
            >
              연락하기
            </a>
          </div>
        </div>

        <aside className="rounded-2xl border border-border bg-surface/95 p-6 shadow-[0_20px_50px_-30px_rgba(28,25,21,0.35)]">
          <div className="mb-4 flex items-center justify-between">
            <p className="section-label">Workbench</p>
            <WorkbenchMark />
          </div>
          <ul className="space-y-3 text-sm leading-relaxed text-muted">
            <li className="flex gap-3">
              <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-accent" />
              결정 로그 · 왜 그렇게 골랐는지
            </li>
            <li className="flex gap-3">
              <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-warm" />
              지식 노트 · 읽기와 정리의 루프
            </li>
            <li className="flex gap-3">
              <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-[#6b7c8a]" />
              봇 워크플로 · 반복을 책상 밖으로
            </li>
          </ul>
          <p className="mt-5 border-t border-border pt-4 font-mono text-xs text-muted">
            Honest lab notes · No vanity metrics
          </p>
        </aside>
      </div>
    </section>
  );
}

/** Abstract desk / workbench motif — no animals or robots */
function WorkbenchMark() {
  return (
    <svg
      width="40"
      height="28"
      viewBox="0 0 40 28"
      fill="none"
      aria-hidden="true"
      className="text-accent/70"
    >
      <rect
        x="2"
        y="16"
        width="36"
        height="3"
        rx="0.5"
        stroke="currentColor"
        strokeWidth="1.25"
      />
      <rect
        x="6"
        y="19"
        width="2"
        height="7"
        stroke="currentColor"
        strokeWidth="1.25"
      />
      <rect
        x="32"
        y="19"
        width="2"
        height="7"
        stroke="currentColor"
        strokeWidth="1.25"
      />
      <rect
        x="8"
        y="6"
        width="14"
        height="10"
        rx="1"
        stroke="currentColor"
        strokeWidth="1.25"
      />
      <line
        x1="10"
        y1="9"
        x2="20"
        y2="9"
        stroke="currentColor"
        strokeWidth="1"
        opacity="0.5"
      />
      <line
        x1="10"
        y1="12"
        x2="18"
        y2="12"
        stroke="currentColor"
        strokeWidth="1"
        opacity="0.5"
      />
      <circle cx="30" cy="11" r="3.5" stroke="currentColor" strokeWidth="1.25" />
      <path
        d="M28.5 11h3M30 9.5v3"
        stroke="currentColor"
        strokeWidth="1"
        strokeLinecap="round"
      />
    </svg>
  );
}
