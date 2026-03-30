/**
 * Print/PDF resume: no site chrome (nav/footer). Light, paper-oriented surface only.
 */
export default function ResumePrintLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="resume-print-root min-h-screen bg-white text-neutral-950 print:bg-white">
      <main className="min-h-screen">{children}</main>
    </div>
  );
}
