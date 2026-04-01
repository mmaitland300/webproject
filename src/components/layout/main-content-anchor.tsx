/** Hash target for the site skip link (`#main-content`). */
export function MainContentAnchor() {
  return (
    <div
      id="main-content"
      tabIndex={-1}
      aria-label="Main content"
      className="scroll-mt-16 h-px w-full shrink-0 overflow-hidden outline-none"
    />
  );
}
