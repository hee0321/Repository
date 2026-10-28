export function About() {
  return (
    <section id="about" className="border-b border-border">
      <div className="mx-auto grid max-w-5xl gap-10 px-5 py-20 sm:px-8 sm:py-24 lg:grid-cols-[0.9fr_1.1fr]">
        <div>
          <p className="section-label mb-4">About</p>
          <h2 className="font-serif text-3xl font-semibold tracking-tight text-ink sm:text-4xl">
            사고의 실험실이란
          </h2>
        </div>
        <div className="space-y-5 text-base leading-relaxed text-muted sm:text-[1.05rem]">
          <p className="margin-rule">
            사피엔스는 포트폴리오 쇼룸이 아닙니다. 노트·결정·봇이 같은 책상
            위에서 섞이는{" "}
            <span className="font-medium text-ink">생각 워크벤치</span>
            입니다. 읽은 것, 고른 이유, 자동화한 반복이 한곳에 남습니다.
          </p>
          <p>
            과장된 지표, 가짜 팀 소개, 포장된 케이스 스터디는 두지 않습니다.
            현희가 실제로 만지는 실험과 솔직한 메모만 올립니다.
          </p>
          <div className="grid gap-3 pt-2 sm:grid-cols-3">
            {[
              { k: "책상 위", v: "노트 · 결정 · 봇" },
              { k: "방식", v: "작은 실험" },
              { k: "태도", v: "따뜻하고 날카롭게" },
            ].map((item) => (
              <div
                key={item.k}
                className="rounded-xl border border-border bg-surface px-4 py-3"
              >
                <p className="section-label mb-1">{item.k}</p>
                <p className="text-sm font-medium text-ink">{item.v}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
