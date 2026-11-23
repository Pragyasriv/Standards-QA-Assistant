import { formatAnswer } from "../utils/formatAnswer";

type Props = {
  answer: string;
};

export function AnswerContent({ answer }: Props) {
  const formatted = formatAnswer(answer);

  if (formatted.kind === "plain") {
    return (
      <pre className="answer-prose whitespace-pre-wrap text-[0.9375rem] leading-relaxed text-slate-800">
        {formatted.text}
      </pre>
    );
  }

  return (
    <div className="answer-prose space-y-5 text-[0.9375rem] leading-relaxed text-slate-800">
      {formatted.intro && (
        <p className="text-slate-700">{formatted.intro}</p>
      )}
      {formatted.entries.length > 0 && (
        <ul className="list-none space-y-2 border-l-2 border-accent/20 pl-4">
          {formatted.entries.map((line) => (
            <li
              key={line}
              className="relative font-medium text-slate-900 before:absolute before:-left-4 before:top-[0.55rem] before:h-1.5 before:w-1.5 before:rounded-full before:bg-accent/70"
            >
              {line}
            </li>
          ))}
        </ul>
      )}
      {formatted.qualifications.length > 0 && (
        <QualificationSection
          title="Applicable conditions and exceptions"
          lines={formatted.qualifications}
        />
      )}
    </div>
  );
}

function QualificationSection({ title, lines }: { title: string; lines: string[] }) {
  return (
    <section className="border-t border-slate-100 pt-4">
      <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
        {title}
      </h4>
      <div className="mt-3 space-y-3">
        {lines.map((line) => (
          <p key={line} className="text-slate-800">
            {line}
          </p>
        ))}
      </div>
    </section>
  );
}