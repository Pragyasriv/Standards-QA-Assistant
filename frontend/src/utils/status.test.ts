import { describe, expect, it } from "vitest";
import { describeAskStatus, isSuccessAnswer } from "./status";

describe("status helpers", () => {
  it("treats only answered as success", () => {
    expect(isSuccessAnswer("answered")).toBe(true);
    expect(isSuccessAnswer("abstained")).toBe(false);
    expect(isSuccessAnswer("insufficient_evidence")).toBe(false);
  });

  it("maps clarification status", () => {
    const info = describeAskStatus("clarification_required");
    expect(info.label).toContain("Clarification");
    expect(info.tone).toBe("info");
  });
});