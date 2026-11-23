import { Loader2 } from "lucide-react";

const STEPS = [
  "Running scope and evidence guardrails",
  "Hybrid retrieval when approved",
  "Generating and validating citations",
];

export function LoadingPanel() {
  return (
    <section
      className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm sm:p-8"
      aria-live="polite"
      aria-busy="true"
    >
      <div className="mx-auto flex max-w-read items-start gap-4">
        <Loader2 className="mt-0.5 h-5 w-5 shrink-0 animate-spin text-accent" />
        <div>
          <h3 className="text-sm font-semibold text-slate-900">Processing your question</h3>
          <p className="mt-1 text-sm text-slate-600">
            The backend may retrieve evidence and call the configured model. This
            typically takes a few seconds.
          </p>
          <ul className="mt-4 space-y-2">
            {STEPS.map((step) => (
              <li
                key={step}
                className="flex items-center gap-2 text-xs text-slate-500 before:h-1 before:w-1 before:rounded-full before:bg-slate-300 before:content-['']"
              >
                {step}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
}