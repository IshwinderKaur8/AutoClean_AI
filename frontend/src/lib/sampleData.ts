/**
 * Deterministic, deliberately messy demo dataset so a first-time user can try
 * the whole flow without hunting for a CSV. Contains: missing values in
 * several columns, exact duplicate rows, numeric outliers, inconsistent
 * category casing, an identifier column, a datetime column and a binary label.
 */
function lcg(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
}

export function buildSampleCsv(rows = 240): string {
  const rand = lcg(42);
  const pick = <T,>(arr: T[]) => arr[Math.floor(rand() * arr.length)];
  const cities = ["Delhi", "Mumbai", "Pune", "Bengaluru", "Chennai", "delhi", "MUMBAI"];
  const plans = ["basic", "pro", "enterprise"];
  const header = ["customer_id", "age", "city", "plan", "monthly_spend", "signup_date", "support_tickets", "churned"];
  const lines: string[] = [header.join(",")];
  const body: string[] = [];

  for (let i = 0; i < rows; i++) {
    const age = Math.round(22 + rand() * 45);
    const plan = pick(plans);
    const base = plan === "basic" ? 20 : plan === "pro" ? 60 : 180;
    let spend = Math.round((base + rand() * base * 0.6) * 100) / 100;
    if (rand() < 0.03) spend = Math.round(spend * 25); // outliers
    const tickets = Math.floor(rand() * rand() * 12);
    const churn = rand() < 0.18 + tickets * 0.04 + (plan === "basic" ? 0.1 : 0) ? "yes" : "no";
    const d = new Date(Date.UTC(2022, 0, 1) + Math.floor(rand() * 900) * 86_400_000);
    const date = d.toISOString().slice(0, 10);

    const cells = [
      `C${String(1000 + i)}`,
      rand() < 0.08 ? "" : String(age),
      rand() < 0.06 ? "" : pick(cities),
      plan,
      rand() < 0.07 ? "" : String(spend),
      rand() < 0.05 ? "" : date,
      String(tickets),
      churn,
    ];
    body.push(cells.join(","));
  }
  // exact duplicates
  for (let k = 0; k < 6; k++) body.push(body[Math.floor(rand() * body.length)]);
  lines.push(...body);
  return lines.join("\n") + "\n";
}

export function sampleFile(): File {
  return new File([buildSampleCsv()], "sample_customers.csv", { type: "text/csv" });
}
