export function Footer() {
  return (
    <footer className="border-t border-border">
      <div className="mx-auto flex max-w-5xl flex-col gap-3 px-5 py-8 text-sm text-muted sm:flex-row sm:items-center sm:justify-between sm:px-8">
        <p>
          <span className="font-serif font-semibold text-ink">사피엔스</span>
          <span className="mx-2 text-border">·</span>
          사고의 실험실
        </p>
        <p className="font-mono text-xs tracking-wide">
          © {new Date().getFullYear()} Sapiens · Thought Laboratory
        </p>
      </div>
    </footer>
  );
}
