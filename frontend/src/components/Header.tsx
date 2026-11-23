type Props = {
  subtitle?: string;
};

export function Header({ subtitle }: Props) {
  return (
    <header className="border-b border-slate-200 bg-white px-4 py-5 sm:px-8">
      <h2 className="text-heading-lg">
        Regulatory Q&amp;A workspace
      </h2>
      <p className="mt-1.5 max-w-2xl text-sm leading-relaxed text-slate-600">
        {subtitle ??
          "Ask questions about RoHS and REACH. The guardrails classify scope and evidence before any answer is generated."}
      </p>
    </header>
  );
}