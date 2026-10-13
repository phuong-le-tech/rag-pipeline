import type { Metadata } from "next";
import { CONTACT, LLM_MODEL, PUBLISHER, UPDATED } from "@/lib/site.ts";

export const metadata: Metadata = { title: "Privacy · Measured RAG" };

export default function Privacy() {
  return (
    <main className="legal">
      <p><a href="/">← Back to the demo</a></p>
      <h1>Privacy</h1>
      <p>This demo is built to keep as little as possible: the app itself stores nothing about you.</p>

      <h2>Who is responsible</h2>
      <p>
        {PUBLISHER || "The publisher named in the legal notice"}
        {CONTACT ? <> (<a href={`mailto:${CONTACT}`}>{CONTACT}</a>)</> : null}, a personal, non-commercial
        demo. See the <a href="/legal">legal notice</a>.
      </p>

      <h2>What is processed</h2>
      <p>
        The question you type is sent to our API to search the Kubernetes documentation, then to a
        large language model to write the answer. The model is <code>{LLM_MODEL}</code>, reached
        through OpenRouter. Free model providers may log prompts.{" "}
        <strong>Do not enter personal or confidential information.</strong>
      </p>

      <h2>What is stored</h2>
      <p>
        The app saves no questions and no answers. Its own logs record a random request id, timings
        and a status code (plus technical error details when something fails), not your question and
        not your IP address. Your IP address is kept in memory for about a minute to limit abuse
        (10 questions per minute), then discarded. The hosting providers below keep their own
        standard access logs (which include IP addresses) under their policies, and the model
        provider may log prompts.
      </p>

      <h2>Legal basis</h2>
      <p>
        Legitimate interest (GDPR art. 6(1)(f)): answering the question you choose to send, and
        protecting the service from abuse with short-lived rate limiting.
      </p>

      <h2>Transfers outside the EU</h2>
      <p>
        The providers below are based in the United States. Your question and technical request data
        are processed there under their standard contractual safeguards.
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
        The app keeps no data about you, so there is nothing for it to access, correct or delete; for
        data held by the providers, use their contact points above.
        {CONTACT ? <> Questions: <a href={`mailto:${CONTACT}`}>{CONTACT}</a>.</> : null} You can also
        complain to your data protection authority (in France, the{" "}
        <a href="https://www.cnil.fr/en/plaintes">CNIL</a>).
      </p>

      <p className="fine">Last updated {UPDATED}.</p>
    </main>
  );
}
