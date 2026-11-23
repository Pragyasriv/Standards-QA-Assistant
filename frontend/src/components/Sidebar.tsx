import { BookOpen, Plus, Scale } from "lucide-react";
import { EXAMPLE_QUESTIONS } from "../constants/examples";
import type { ApiStatusState } from "../hooks/useApiStatus";
import { ExampleQuestions } from "./ExampleQuestions";
import { StatusIndicator } from "./StatusIndicator";

type Props = {
  apiStatus: ApiStatusState;
  onNewQuestion: () => void;
  onSelectExample: (question: string) => void;
  disabled?: boolean;
};

export function Sidebar({
  apiStatus,
  onNewQuestion,
  onSelectExample,
  disabled,
}: Props) {
  return (
    <aside className="flex h-full w-full flex-col border-r border-slate-200 bg-[#0f1c33] text-slate-100 lg:w-72 lg:shrink-0">
      <div className="border-b border-white/10 p-5">
        <div className="flex items-start gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-white/10">
            <Scale className="h-5 w-5 text-sky-200" aria-hidden />
          </div>
          <div>
            <h1 className="text-base font-semibold tracking-tight text-white">
              Standards Q&amp;A Assistant
            </h1>
            <p className="mt-1 text-xs text-slate-300">
              Citation-backed regulatory intelligence
            </p>
          </div>
        </div>
      </div>

      <div className="flex flex-1 flex-col gap-5 overflow-y-auto p-4">
        <button
          type="button"
          onClick={onNewQuestion}
          disabled={disabled}
          className="flex w-full items-center justify-center gap-2 rounded-lg border border-white/15 bg-white/10 px-3 py-2.5 text-sm font-semibold text-white transition hover:bg-white/15 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <Plus className="h-4 w-4" />
          New question
        </button>

        <div>
          <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
            Example questions
          </p>
          <ExampleQuestions
            questions={[...EXAMPLE_QUESTIONS]}
            onSelect={onSelectExample}
            variant="sidebar"
            disabled={disabled}
          />
        </div>

        <div className="rounded-lg border border-white/10 bg-white/5 p-3 text-xs leading-relaxed text-slate-300">
          <div className="mb-2 flex items-center gap-2 font-medium text-slate-200">
            <BookOpen className="h-4 w-4" />
            Supported regulations
          </div>
          <p>
            Official consolidated texts for{" "}
            <span className="text-white">RoHS</span> and{" "}
            <span className="text-white">REACH</span>. Answers are grounded in
            retrieved evidence with server-resolved citations—not legal advice.
          </p>
        </div>

        <div className="mt-auto">
          <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
            System status
          </p>
          <div className="[&_*]:text-slate-800">
            <StatusIndicator status={apiStatus} />
          </div>
        </div>
      </div>
    </aside>
  );
}