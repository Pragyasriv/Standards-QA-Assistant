import { describe, expect, it } from "vitest";
import { formatAnswer } from "./formatAnswer";

const ENTRIES = [
  "Lead (0,1 %)",
  "Mercury (0,1 %)",
  "Cadmium (0,01 %)",
];

const CONDITIONS =
  "The restriction of DEHP, BBP, DBP and DIBP shall apply to medical devices from 22 July 2021.";

describe("formatAnswer", () => {
  it("structures contiguous entry lines and qualification paragraphs", () => {
    const answer =
      "The restricted substances listed in the supplied text are:\n" +
      ENTRIES.join("\n") +
      "\n" +
      CONDITIONS;
    const result = formatAnswer(answer);
    expect(result.kind).toBe("structured");
    if (result.kind === "structured") {
      expect(result.entries).toEqual(ENTRIES);
      expect(result.qualifications).toEqual([CONDITIONS]);
      expect(result.intro).toContain("restricted substances");
    }
  });

  it("returns plain text for narrative answers", () => {
    const answer = "Article 4(1) prohibits placing non-compliant EEE on the market.";
    expect(formatAnswer(answer)).toEqual({ kind: "plain", text: answer });
  });

  it("returns plain text when entries are not contiguous", () => {
    const answer = "Lead (0,1 %)\n\nIntro line\nMercury (0,1 %)";
    expect(formatAnswer(answer).kind).toBe("plain");
  });

  it("preserves exact entry strings", () => {
    const line = "Bis(2-ethylhexyl) phthalate (DEHP) (0,1 %)";
    const answer = `${line}\n${CONDITIONS}`;
    const result = formatAnswer(answer);
    expect(result.kind).toBe("structured");
    if (result.kind === "structured") {
      expect(result.entries[0]).toBe(line);
      expect(result.qualifications[0]).toBe(CONDITIONS);
    }
  });
});