import { ShieldCheck } from "lucide-react";
import { EXAMPLE_QUESTIONS } from "../constants/examples";
import { ExampleQuestions } from "./ExampleQuestions";
import { WorkflowSteps } from "./WorkflowSteps";

type Props = {
  onSelectExample: (question: string) => void;
  disabled?: boolean;
};

export function WelcomePanel({ onSelectExample, disabled }: Props) {
  return (
    <section className="panel-elevated p-6 sm:p-8">
      <div className="flex items-start gap-4">
        <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-lg bg-accent/10">
          <ShieldCheck className="h-6 w-6 text-accent" aria-hidden />
        </div>
        <div className="min-w-0">
          <h3 className="text-heading-md text-slate-900">
            Ask with evidence, not guesswork
          </h3>
          <p className="mt-2 max-w-3xl text-sm leading-relaxed text-slate-600">
            RoHS and REACH questions are classified, retrieved from the local index,
            and answered only when cited chunks support the response. Validation
            failures and guardrail rejections do not expose unverified answers or
            citations.
          </p>
        </div>
      </div>

      <div className="mt-8 min-w-0">
        <p className="text-label mb-3">Try an example</p>
        <ExampleQuestions
          questions={[...EXAMPLE_QUESTIONS]}
          onSelect={onSelectExample}
          disabled={disabled}
          variant="main"
        />
      </div>

      <div className="mt-8 min-w-0 border-t border-slate-100 pt-6">
        <p className="text-label mb-3">How a question is processed</p>
        <WorkflowSteps />
      </div>
    </section>
  );
}