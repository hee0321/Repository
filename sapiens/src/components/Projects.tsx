const projects = [
  {
    id: "decisions",
    tag: "Decisions",
    title: "결정 로그",
    status: "샘플 · 개념",
    description:
      "선택의 근거와 트레이드오프를 짧게 남기는 로그. 나중에 ‘왜 그랬지?’를 다시 열어볼 수 있게, 감정보다 이유를 먼저 적습니다.",
    note: "샘플 카드 — 공개 로그는 준비 중입니다.",
  },
  {
    id: "notes",
    tag: "Knowledge",
    title: "지식 노트",
    status: "샘플 · 개념",
    description:
      "읽기·대화·실험에서 나온 조각을 모으는 개인 지식 노트. 주제는 생각의 도구, 주의력, 일과 학습의 루프에 가깝습니다.",
    note: "샘플 카드 — 공개 노트는 준비 중입니다.",
  },
  {
    id: "workflows",
    tag: "Automation",
    title: "봇 워크플로 · 실험 일지",
    status: "샘플 · 개념",
    description:
      "반복되는 정리·알림·초안을 봇과 스크립트로 책상 밖으로 보내는 실험. 거창한 플랫폼이 아니라, 일상에 붙는 작은 워크플로와 그 일지.",
    note: "샘플 카드 — 구체 사례는 추후 공개 예정입니다.",
  },
];

export function Projects() {
  return (
    <section id="projects" className="border-b border-border bg-surface-2/40">
      <div className="mx-auto max-w-5xl px-5 py-20 sm:px-8 sm:py-24">
        <div className="mb-12 max-w-2xl">
          <p className="section-label mb-4">Workbench</p>
          <h2 className="font-serif text-3xl font-semibold tracking-tight text-ink sm:text-4xl">
            실험대 위의 것들
          </h2>
          <p className="mt-4 text-base leading-relaxed text-muted">
            완료된 케이스 스터디가 아니라, 책상에 올려 두고 만지는 세 가지
            자리입니다. 아래 카드는 개념 샘플입니다.
          </p>
        </div>

        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {projects.map((project, index) => (
            <article
              key={project.id}
              className="flex flex-col rounded-2xl border border-border bg-surface p-6 transition hover:border-[#cfc4b2]"
            >
              <div className="mb-4 flex h-24 items-end justify-center rounded-xl border border-dashed border-border bg-background/80 desk-grid">
                <CardMotif index={index} />
              </div>
              <div className="mb-3 flex flex-wrap items-center gap-2">
                <span className="rounded-full bg-accent-soft px-2.5 py-1 font-mono text-[11px] tracking-wide text-accent">
                  {project.tag}
                </span>
                <span className="rounded-full border border-border px-2.5 py-1 font-mono text-[11px] text-muted">
                  {project.status}
                </span>
              </div>
              <h3 className="font-serif text-xl font-semibold text-ink">
                {project.title}
              </h3>
              <p className="mt-3 flex-1 text-sm leading-relaxed text-muted">
                {project.description}
              </p>
              <p className="mt-4 text-xs text-warm">{project.note}</p>
            </article>
          ))}
        </div>
      </div>
    </section>
  );
}

function CardMotif({ index }: { index: number }) {
  if (index === 0) {
    // Decision log — checklist / ledger lines
    return (
      <svg width="72" height="56" viewBox="0 0 72 56" fill="none" aria-hidden className="text-accent/60 mb-2">
        <rect x="12" y="8" width="48" height="40" rx="2" stroke="currentColor" strokeWidth="1.5" />
        <path d="M20 20h8M32 20h20M20 28h8M32 28h16M20 36h8M32 36h12" stroke="currentColor" strokeWidth="1.25" strokeLinecap="round" />
        <path d="M21 19.5l2 2 4-4" stroke="currentColor" strokeWidth="1.25" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    );
  }
  if (index === 1) {
    // Knowledge notes — stacked pages
    return (
      <svg width="72" height="56" viewBox="0 0 72 56" fill="none" aria-hidden className="text-warm/70 mb-2">
        <rect x="18" y="14" width="36" height="30" rx="1.5" stroke="currentColor" strokeWidth="1.5" />
        <rect x="14" y="10" width="36" height="30" rx="1.5" stroke="currentColor" strokeWidth="1.5" fill="var(--surface)" />
        <path d="M22 20h20M22 26h16M22 32h12" stroke="currentColor" strokeWidth="1.25" strokeLinecap="round" opacity="0.7" />
      </svg>
    );
  }
  // Bot workflow — connected nodes (abstract, not a robot)
  return (
    <svg width="72" height="56" viewBox="0 0 72 56" fill="none" aria-hidden className="text-[#6b7c8a] mb-2">
      <circle cx="18" cy="28" r="6" stroke="currentColor" strokeWidth="1.5" />
      <circle cx="36" cy="18" r="6" stroke="currentColor" strokeWidth="1.5" />
      <circle cx="36" cy="38" r="6" stroke="currentColor" strokeWidth="1.5" />
      <circle cx="54" cy="28" r="6" stroke="currentColor" strokeWidth="1.5" />
      <path d="M24 28h6M42 22l6 4M42 34l6-4M36 24v8" stroke="currentColor" strokeWidth="1.25" strokeLinecap="round" />
    </svg>
  );
}
