#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import process from "node:process";

const DEFAULT_MODELS = ["auto", "opus-4.6"];
const DEFAULT_PROMPTS_PATH = path.join(process.cwd(), "scripts", "model-cost-prompts.json");
const DEFAULT_OUTPUT_CSV = path.join(process.cwd(), "scripts", "model-cost-results.csv");

/**
 * Environment variables:
 * - LLM_API_KEY                (required)
 * - LLM_API_BASE               (optional, default: https://api.openai.com/v1)
 * - LLM_API_PATH               (optional, default: /responses)
 * - LLM_AUTH_SCHEME            (optional, default: Bearer)
 * - COMPARE_MODELS             (optional, comma list; default: auto,opus-4.6)
 * - COMPARE_PROMPTS_PATH       (optional)
 * - COMPARE_OUTPUT_CSV         (optional)
 * - COMPARE_ITERATIONS         (optional, default: 1)
 * - COMPARE_TEMPERATURE        (optional, default: 0)
 * - COMPARE_MAX_OUTPUT_TOKENS  (optional, default: 500)
 * - COMPARE_INCLUDE_REASONING  (optional, default: false)
 */

function parseNumber(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function nowIso() {
  return new Date().toISOString();
}

function csvEscape(value) {
  const raw = String(value ?? "");
  if (raw.includes(",") || raw.includes('"') || raw.includes("\n")) {
    return `"${raw.replaceAll('"', '""')}"`;
  }
  return raw;
}

function getUsage(responseJson) {
  const usage = responseJson?.usage ?? {};
  const inputTokens =
    usage.input_tokens ??
    usage.prompt_tokens ??
    usage.inputTokens ??
    0;
  const outputTokens =
    usage.output_tokens ??
    usage.completion_tokens ??
    usage.outputTokens ??
    0;
  const totalTokens =
    usage.total_tokens ??
    usage.totalTokens ??
    inputTokens + outputTokens;
  return { inputTokens, outputTokens, totalTokens };
}

function getText(responseJson) {
  if (typeof responseJson?.output_text === "string") {
    return responseJson.output_text;
  }
  if (Array.isArray(responseJson?.output)) {
    const chunks = [];
    for (const item of responseJson.output) {
      if (Array.isArray(item?.content)) {
        for (const c of item.content) {
          if (typeof c?.text === "string") {
            chunks.push(c.text);
          }
        }
      }
    }
    if (chunks.length > 0) return chunks.join("\n");
  }
  return "";
}

function estimateCostUsd({ inputTokens, outputTokens }, pricing) {
  if (!pricing) return null;
  const inCost = (inputTokens / 1_000_000) * pricing.inputPer1M;
  const outCost = (outputTokens / 1_000_000) * pricing.outputPer1M;
  return inCost + outCost;
}

function readPrompts(filePath) {
  const raw = fs.readFileSync(filePath, "utf8");
  const parsed = JSON.parse(raw);
  if (!Array.isArray(parsed) || parsed.length === 0) {
    throw new Error(`Prompts file must be a non-empty JSON array: ${filePath}`);
  }
  return parsed.map((x, i) => {
    if (typeof x !== "string" || x.trim() === "") {
      throw new Error(`Prompt at index ${i} must be a non-empty string`);
    }
    return x;
  });
}

function readPricing(pathFromEnv) {
  if (!pathFromEnv) return null;
  const abs = path.isAbsolute(pathFromEnv)
    ? pathFromEnv
    : path.join(process.cwd(), pathFromEnv);
  if (!fs.existsSync(abs)) {
    throw new Error(`Pricing file not found: ${abs}`);
  }
  return JSON.parse(fs.readFileSync(abs, "utf8"));
}

async function callModel({
  apiBase,
  apiPath,
  authScheme,
  apiKey,
  model,
  prompt,
  temperature,
  maxOutputTokens,
  includeReasoning,
}) {
  const url = `${apiBase.replace(/\/$/, "")}${apiPath}`;
  const body = {
    model,
    input: prompt,
    temperature,
    max_output_tokens: maxOutputTokens,
  };

  if (includeReasoning) {
    body.reasoning = { effort: "medium" };
  }

  const started = Date.now();
  const res = await fetch(url, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `${authScheme} ${apiKey}`,
    },
    body: JSON.stringify(body),
  });
  const latencyMs = Date.now() - started;

  if (!res.ok) {
    const details = await res.text();
    throw new Error(`HTTP ${res.status} from ${url}: ${details}`);
  }

  const json = await res.json();
  const usage = getUsage(json);
  const text = getText(json);

  return {
    modelRequested: model,
    modelServed: json?.model ?? model,
    usage,
    latencyMs,
    outputChars: text.length,
  };
}

function summarize(rows) {
  const byModel = new Map();
  for (const row of rows) {
    const key = row.modelServed;
    const prev = byModel.get(key) ?? {
      requests: 0,
      inputTokens: 0,
      outputTokens: 0,
      totalTokens: 0,
      latencyMs: 0,
      costUsd: 0,
      costKnown: true,
    };
    prev.requests += 1;
    prev.inputTokens += row.inputTokens;
    prev.outputTokens += row.outputTokens;
    prev.totalTokens += row.totalTokens;
    prev.latencyMs += row.latencyMs;
    if (row.costUsd == null) {
      prev.costKnown = false;
    } else {
      prev.costUsd += row.costUsd;
    }
    byModel.set(key, prev);
  }
  return byModel;
}

function printSummary(byModel) {
  console.log("\n=== Summary ===");
  for (const [model, s] of byModel.entries()) {
    const avgLatency = s.requests ? s.latencyMs / s.requests : 0;
    console.log(`\nModel: ${model}`);
    console.log(`  Requests:      ${s.requests}`);
    console.log(`  Input tokens:  ${s.inputTokens}`);
    console.log(`  Output tokens: ${s.outputTokens}`);
    console.log(`  Total tokens:  ${s.totalTokens}`);
    console.log(`  Avg latency:   ${avgLatency.toFixed(1)} ms`);
    if (s.costKnown) {
      console.log(`  Est. cost:     $${s.costUsd.toFixed(6)}`);
    } else {
      console.log("  Est. cost:     n/a (no pricing loaded)");
    }
  }
}

async function main() {
  const apiKey = process.env.LLM_API_KEY;
  if (!apiKey) {
    throw new Error("Missing LLM_API_KEY");
  }

  const apiBase = process.env.LLM_API_BASE || "https://api.openai.com/v1";
  const apiPath = process.env.LLM_API_PATH || "/responses";
  const authScheme = process.env.LLM_AUTH_SCHEME || "Bearer";
  const models = (process.env.COMPARE_MODELS || DEFAULT_MODELS.join(","))
    .split(",")
    .map((m) => m.trim())
    .filter(Boolean);
  const promptsPath = process.env.COMPARE_PROMPTS_PATH || DEFAULT_PROMPTS_PATH;
  const outputCsv = process.env.COMPARE_OUTPUT_CSV || DEFAULT_OUTPUT_CSV;
  const iterations = Math.max(1, parseNumber(process.env.COMPARE_ITERATIONS, 1));
  const temperature = parseNumber(process.env.COMPARE_TEMPERATURE, 0);
  const maxOutputTokens = Math.max(
    1,
    parseNumber(process.env.COMPARE_MAX_OUTPUT_TOKENS, 500)
  );
  const includeReasoning = String(process.env.COMPARE_INCLUDE_REASONING || "false")
    .toLowerCase()
    .trim() === "true";
  const pricing = readPricing(process.env.COMPARE_PRICING_JSON || "");
  const prompts = readPrompts(promptsPath);

  const rows = [];
  const header = [
    "timestamp",
    "iteration",
    "promptIndex",
    "modelRequested",
    "modelServed",
    "inputTokens",
    "outputTokens",
    "totalTokens",
    "latencyMs",
    "costUsd",
  ];

  for (let iter = 1; iter <= iterations; iter += 1) {
    for (let p = 0; p < prompts.length; p += 1) {
      const prompt = prompts[p];
      for (const model of models) {
        const result = await callModel({
          apiBase,
          apiPath,
          authScheme,
          apiKey,
          model,
          prompt,
          temperature,
          maxOutputTokens,
          includeReasoning,
        });
        const costUsd = estimateCostUsd(result.usage, pricing?.[result.modelServed] || pricing?.[model]);
        rows.push({
          timestamp: nowIso(),
          iteration: iter,
          promptIndex: p,
          modelRequested: result.modelRequested,
          modelServed: result.modelServed,
          inputTokens: result.usage.inputTokens,
          outputTokens: result.usage.outputTokens,
          totalTokens: result.usage.totalTokens,
          latencyMs: result.latencyMs,
          costUsd,
        });
        console.log(
          `[iter ${iter}] prompt ${p + 1}/${prompts.length} | requested=${result.modelRequested} served=${result.modelServed} | in=${result.usage.inputTokens} out=${result.usage.outputTokens} | ${result.latencyMs}ms`
        );
      }
    }
  }

  const lines = [header.join(",")];
  for (const row of rows) {
    lines.push(
      [
        row.timestamp,
        row.iteration,
        row.promptIndex,
        row.modelRequested,
        row.modelServed,
        row.inputTokens,
        row.outputTokens,
        row.totalTokens,
        row.latencyMs,
        row.costUsd == null ? "" : row.costUsd.toFixed(8),
      ]
        .map(csvEscape)
        .join(",")
    );
  }
  fs.mkdirSync(path.dirname(outputCsv), { recursive: true });
  fs.writeFileSync(outputCsv, `${lines.join("\n")}\n`, "utf8");

  const summary = summarize(rows);
  printSummary(summary);
  console.log(`\nCSV written to: ${outputCsv}`);
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
