/** Split grounded answers for display only. Wording is never altered. */

export type PlainAnswer = {
  kind: "plain";
  text: string;
};

export type StructuredAnswer = {
  kind: "structured";
  intro: string | null;
  entries: string[];
  qualifications: string[];
};

export type FormattedAnswer = PlainAnswer | StructuredAnswer;

/** RoHS-style substance lines with decimal-comma limits, e.g. Lead (0,1 %) */
const ENTRY_LINE =
  /^\s*.+\(\d+,\d+\s*%\)\s*$|^\s*.+\(\d+\.\d+\s*%\)\s*$/;

const QUALIFICATION_MARKER = /\bshall (not )?apply\b/i;

function isEntryLine(line: string): boolean {
  return ENTRY_LINE.test(line);
}

function isContiguous(indices: number[]): boolean {
  if (indices.length === 0) return false;
  for (let i = 1; i < indices.length; i++) {
    if (indices[i] !== indices[i - 1] + 1) return false;
  }
  return true;
}

function joinLines(lines: string[]): string {
  return lines.join("\n");
}

/**
 * Infer list vs qualification sections when the pattern matches backend fidelity
 * entry lines. Falls back to the full original string when unsure.
 */
export function formatAnswer(answer: string): FormattedAnswer {
  if (!answer.trim()) {
    return { kind: "plain", text: answer };
  }

  const lines = answer.split("\n");
  const entryIndices: number[] = [];
  for (let i = 0; i < lines.length; i++) {
    if (isEntryLine(lines[i])) {
      entryIndices.push(i);
    }
  }

  if (entryIndices.length === 0) {
    return { kind: "plain", text: answer };
  }

  if (!isContiguous(entryIndices)) {
    return { kind: "plain", text: answer };
  }

  const firstEntry = entryIndices[0];
  const lastEntry = entryIndices[entryIndices.length - 1];

  const introLines = lines.slice(0, firstEntry).filter((line) => line.trim() !== "");
  const entries = entryIndices.map((i) => lines[i]);
  const tailLines = lines.slice(lastEntry + 1).filter((line) => line.trim() !== "");
  const qualificationLines = tailLines.filter((line) => !isEntryLine(line));

  const hasQualMarker = qualificationLines.some((line) =>
    QUALIFICATION_MARKER.test(line),
  );

  if (entryIndices.length < 2 && !(entryIndices.length === 1 && hasQualMarker)) {
    return { kind: "plain", text: answer };
  }

  if (qualificationLines.length === 0 && entryIndices.length < 3) {
    // Short answers with only entries—still show as structured list if multiple entries
    if (entryIndices.length < 2) {
      return { kind: "plain", text: answer };
    }
  }

  return {
    kind: "structured",
    intro: introLines.length ? joinLines(introLines) : null,
    entries,
    qualifications: qualificationLines,
  };
}