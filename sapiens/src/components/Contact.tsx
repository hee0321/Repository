export function Contact() {
  return (
    <section id="contact" className="border-b border-border">
      <div className="mx-auto max-w-5xl px-5 py-20 sm:px-8 sm:py-24">
        <div className="rounded-2xl border border-border bg-accent px-6 py-12 text-white sm:px-10 sm:py-14">
          <p className="mb-4 font-mono text-xs tracking-[0.14em] uppercase text-white/70">
            Contact
          </p>
          <h2 className="font-serif text-3xl font-semibold tracking-tight sm:text-4xl">
            책상에 앉아 이야기해 볼까요
          </h2>
          <p className="mt-4 max-w-xl text-base leading-relaxed text-white/80">
            비슷한 실험을 하고 있거나, 노트·결정·봇 워크플로에 대해 이야기하고
            싶은 분이면 편하게 메일을 보내 주세요. (메일 주소는 스텁입니다 —
            실제 주소로 바꿔 쓰세요.)
          </p>
          <a
            href="mailto:hello@sapiens.local"
            className="mt-8 inline-flex h-11 items-center justify-center rounded-full bg-white px-6 text-sm font-medium text-accent transition hover:bg-[#f3f7f5]"
          >
            hello@sapiens.local
          </a>
        </div>
      </div>
    </section>
  );
}
