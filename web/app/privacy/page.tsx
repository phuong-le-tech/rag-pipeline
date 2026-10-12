import type { Metadata } from "next";
import { CONTACT, LLM_MODEL, UPDATED } from "@/lib/site.ts";

export const metadata: Metadata = { title: "Privacy · Glass-box RAG" };

export default function Privacy() {
  return (
    <main className="legal">
      <p><a href="/">← Back to the demo</a></p>
      <h1>Privacy</h1>
      <p>This demo is built to keep as little as possible: it stores nothing about you.</p>

      <h2>What is processed</h2>
      <p>
        The question you type is sent to our API to search the Kubernetes documentation, then to a
        large language model to write the answer. The model is <code>{LLM_MODEL}</code>, reached
        through OpenRouter. Free model providers may log prompts.{" "}
        <strong>Do not enter personal or confidential information.</strong>
      </p>

      <h2>What is stored</h2>
      <p>
        Nothing. Questions and answers are not saved. Server logs record only a random request id,
        timings and a status code: not your question and not your IP address. Your IP address is
        kept in memory for one minute to limit abuse (10 questions per minute), then discarded.
      </p>

      <h2>Browser storage</h2>
      <p>
        Only your light or dark theme choice, in <code>localStorage</code> (key <code>gb-theme</code>).
        No cookies, no analytics, no trackers.
      </p>

      <h2>Service providers</h2>
      <ul>
        <li>Vercel: hosts this website (<a href="https://vercel.com/legal/privacy-policy">privacy policy</a>)</li>
        <li>Railway: hosts the API and the database (<a href="https://railway.com/legal/privacy">privacy policy</a>)</li>
        <li>OpenRouter and the model provider it routes to: generate the answer (<a href="https://openrouter.ai/privacy">privacy policy</a>)</li>
      </ul>

      <h2>Your rights</h2>
      <p>
        Because nothing is stored, there is nothing to access, correct or delete.
        {CONTACT ? <> Questions: <a href={`mailto:${CONTACT}`}>{CONTACT}</a>.</> : null}
      </p>

      <p className="fine">Last updated {UPDATED}.</p>
    </main>
  );
}
