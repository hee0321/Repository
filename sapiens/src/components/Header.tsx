const links = [
  { href: "#about", label: "소개" },
  { href: "#projects", label: "실험대" },
  { href: "#contact", label: "연락" },
];

export function Header() {
  return (
    <header className="sticky top-0 z-50 border-b border-border/80 bg-background/85 backdrop-blur-md">
      <div className="mx-auto flex h-14 max-w-5xl items-center justify-between px-5 sm:px-8">
        <a
          href="#top"
          className="font-serif text-lg font-semibold tracking-tight text-ink"
        >
          사피엔스
          <span className="ml-2 font-sans text-xs font-medium tracking-wide text-muted">
            · 사고의 실험실
          </span>
        </a>
        <nav className="flex items-center gap-1 sm:gap-2" aria-label="주요 메뉴">
          {links.map((link) => (
            <a
              key={link.href}
              href={link.href}
              className="rounded-full px-3 py-1.5 text-sm text-muted transition-colors hover:bg-surface-2 hover:text-ink"
            >
              {link.label}
            </a>
          ))}
        </nav>
      </div>
    </header>
  );
}
