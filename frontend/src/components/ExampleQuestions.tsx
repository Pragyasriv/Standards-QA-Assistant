import { ArrowRight } from "lucide-react";

type Props = {
  questions: string[];
  onSelect: (question: string) => void;
  variant?: "sidebar" | "main";
  disabled?: boolean;
};

export function ExampleQuestions({
  questions,
  onSelect,
  variant = "main",
  disabled,
}: Props) {
  if (variant === "sidebar") {
    return (
      <ul className="space-y-2">
        {questions.map((q) => (
          <li key={q}>
            <button
              type="button"
              disabled={disabled}
              onClick={() => onSelect(q)}
              className="w-full rounded-md border border-white/10 bg-white/5 px-3 py-2 text-left text-xs leading-snug text-slate-200 transition hover:bg-white/10 disabled:opacity-50"
            >
              {q}
            </button>
          </li>
        ))}
      </ul>
    );
  }

  return (
    <ul className="grid min-w-0 grid-cols-1 gap-3 sm:grid-cols-2">
      {questions.map((q, index) => (
        <li key={q} className="min-w-0">
          <button
            type="button"
            disabled={disabled}
            onClick={() => onSelect(q)}
            className="group flex h-full w-full min-w-0 items-start gap-3 rounded-lg border border-slate-200 bg-white px-4 py-3.5 text-left shadow-sm transition hover:border-accent/30 hover:shadow-md disabled:opacity-50"
          >
            <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-slate-100 text-xs font-semibold text-slate-600 group-hover:bg-accent/10 group-hover:text-accent">
              {index + 1}
            </span>
            <span className="min-w-0 flex-1 text-sm leading-snug text-slate-800">
              {q}
              <span className="mt-2 flex items-center gap-1 text-xs font-medium text-accent opacity-0 transition group-hover:opacity-100">
                Use this question
                <ArrowRight className="h-3 w-3 shrink-0" />
              </span>
            </span>
          </button>
        </li>
      ))}
    </ul>
  );
}