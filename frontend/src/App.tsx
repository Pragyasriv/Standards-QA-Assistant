import { useCallback, useState } from "react";
import { AnswerPanel } from "./components/AnswerPanel";
import { Header } from "./components/Header";
import { QuestionInput } from "./components/QuestionInput";
import { Sidebar } from "./components/Sidebar";
import { WelcomePanel } from "./components/WelcomePanel";
import { useApiStatus } from "./hooks/useApiStatus";
import { useAsk } from "./hooks/useAsk";

export default function App() {
  const apiStatus = useApiStatus();
  const ask = useAsk();
  const [question, setQuestion] = useState("");

  const handleNewQuestion = useCallback(() => {
    setQuestion("");
    ask.reset();
  }, [ask]);

  const handleSubmit = useCallback(() => {
    void ask.submit(question);
  }, [ask, question]);

  const handleSelectExample = useCallback(
    (value: string) => {
      setQuestion(value);
      ask.reset();
    },
    [ask],
  );

  const showWelcome = !ask.response && !ask.error && !ask.loading;

  return (
    <div className="flex min-h-full flex-col lg:flex-row">
      <Sidebar
        apiStatus={apiStatus}
        onNewQuestion={handleNewQuestion}
        onSelectExample={handleSelectExample}
        disabled={ask.loading}
      />
      <main className="flex min-h-0 min-w-0 flex-1 flex-col">
        <Header />
        <div className="workspace-shell flex flex-1 flex-col gap-5 overflow-y-auto sm:gap-6">
          <QuestionInput
            value={question}
            onChange={setQuestion}
            onSubmit={handleSubmit}
            loading={ask.loading}
            prominent={showWelcome}
          />

          {showWelcome && (
            <WelcomePanel onSelectExample={handleSelectExample} disabled={ask.loading} />
          )}

          <AnswerPanel
            response={ask.response}
            error={ask.error}
            loading={ask.loading}
          />
        </div>
      </main>
    </div>
  );
}