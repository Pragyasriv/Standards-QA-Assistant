import {
  BookCheck,
  ChevronRight,
  FileQuestion,
  GitMerge,
  Shield,
  Sparkles,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";

const STEPS: { label: string; shortLabel: string; icon: LucideIcon }[] = [
  { label: "Question", shortLabel: "Question", icon: FileQuestion },
  { label: "Guardrails", shortLabel: "Guardrails", icon: Shield },
  { label: "Hybrid retrieval", shortLabel: "Retrieval", icon: GitMerge },
  { label: "Evidence validation", shortLabel: "Validation", icon: BookCheck },
  { label: "Cited answer", shortLabel: "Answer", icon: Sparkles },
];

export function WorkflowSteps() {
  return (
    <div className="min-w-0">
      <ol className="flex w-full min-w-0 flex-nowrap items-center justify-start gap-1 overflow-x-auto pb-0.5 [-ms-overflow-style:none] [scrollbar-width:none] lg:justify-between lg:gap-2 lg:overflow-x-visible [&::-webkit-scrollbar]:hidden">
        {STEPS.map((step, index) => (
          <li key={step.label} className="flex shrink-0 items-center gap-1 lg:gap-2">
            <span
              className="inline-flex items-center gap-1 rounded-md border border-slate-200 bg-slate-50 px-2 py-1.5 text-[11px] font-medium text-slate-700 whitespace-nowrap sm:gap-1.5 sm:px-2.5 sm:text-xs"
              title={step.label}
            >
              <step.icon className="h-3.5 w-3.5 shrink-0 text-accent" aria-hidden />
              <span className="hidden md:inline">{step.label}</span>
              <span className="md:hidden">{step.shortLabel}</span>
            </span>
            {index < STEPS.length - 1 && (
              <ChevronRight
                className="h-3.5 w-3.5 shrink-0 text-slate-300"
                aria-hidden
              />
            )}
          </li>
        ))}
      </ol>
    </div>
  );
}