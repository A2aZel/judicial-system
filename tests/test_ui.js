"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const ui = require("../ui/app.js");

const originalFetch = global.fetch;
const originalWindow = global.window;
const originalSessionStorage = global.sessionStorage;

function numericSummary(count = 0) {
  return { count, median: null, q1: null, q3: null, iqr: null, min: null, max: null };
}

function booleanSummary() {
  return { count: 0, true_count: 0, rate: null };
}

function unavailableRecommendation() {
  return {
    status: "unavailable",
    kind: null,
    execution: null,
    suggested_value: null,
    range_min: null,
    range_max: null,
    unit: null,
    matching_cases: 1,
    usable_outcomes: 1,
    selected_outcomes: 0,
    alternatives: [],
    data_source: "case_corpus",
    reasons: ["Недостаточно наблюдений"],
    legal_status: "not_checked"
  };
}

function validAnalysis() {
  return {
    analysis_id: "analysis-1",
    legal_checks: [{ code: "article", status: "passed", message: "Проверено" }],
    analogues: [{
      distance: 0.25,
      prison_months: 12.5,
      fine_rub: null,
      fine_multiple: true,
      ban_on_positions: true,
      ban_years: null,
      conditional: false,
      confiscation: false
    }],
    punishment_summary: {
      candidate_count: 1,
      returned_count: 1,
      prison_months: { count: 1, median: 12.5, q1: 12.5, q3: 12.5, iqr: 0, min: 12.5, max: 12.5 },
      fine_rub: numericSummary(),
      ban_years: numericSummary(),
      conditional: booleanSummary(),
      confiscation: booleanSummary()
    },
    appeal_risk: { status: "unavailable", score: null, reasons: ["Нет checkpoint"], model: null },
    recommendation: unavailableRecommendation(),
    warnings: [],
    model_info: { engine_version: "1.0.0", knn: { ready: true }, bert: { status: "unavailable" } }
  };
}

function mockResponse({ status = 200, contentType = "application/json", json, text = "" }) {
  return {
    status,
    ok: status >= 200 && status < 300,
    headers: { get: (name) => name.toLowerCase() === "content-type" ? contentType : null },
    json: json || (async () => validAnalysis()),
    text: async () => text
  };
}

test.beforeEach(() => {
  global.window = globalThis;
  global.sessionStorage = {
    getItem: () => null,
    setItem: () => {},
    removeItem: () => {}
  };
});

test.afterEach(() => {
  global.fetch = originalFetch;
  global.window = originalWindow;
  global.sessionStorage = originalSessionStorage;
});

test("AnalysisResponse: пустой объект и неполные структуры отклоняются", () => {
  for (const payload of [null, [], {}, { analysis_id: "id" }, "HTML"]) {
    assert.throws(
      () => ui.validateAnalysisResponse(payload),
      (error) => error.code === "INVALID_ANALYSIS_RESPONSE" && /Результаты не показаны/.test(error.message)
    );
  }
});

test("AnalysisResponse: валидный ответ принимается, невалидный риск отклоняется", () => {
  const response = validAnalysis();
  assert.equal(ui.validateAnalysisResponse(response), response);

  const missingScore = validAnalysis();
  missingScore.appeal_risk = { status: "available", score: null, reasons: [], model: null };
  assert.throws(() => ui.validateAnalysisResponse(missingScore), /appeal_risk\.score/);

  const fabricatedUnavailableScore = validAnalysis();
  fabricatedUnavailableScore.appeal_risk.score = 0.8;
  assert.throws(() => ui.validateAnalysisResponse(fabricatedUnavailableScore), /unavailable/);

  const unranked = validAnalysis();
  unranked.analogues[0].distance = null;
  assert.equal(ui.validateAnalysisResponse(unranked), unranked);

  const numericString = validAnalysis();
  numericString.analogues[0].distance = "0.25";
  assert.throws(() => ui.validateAnalysisResponse(numericString), /analogues\[0\]\.distance/);
});

test("AnalysisResponse: нужна рекомендация API даже при наличии сводки", () => {
  const response = validAnalysis();
  delete response.recommendation;
  assert.throws(() => ui.validateAnalysisResponse(response), /recommendation/);
});

test("AnalysisResponse: недоступная рекомендация не содержит вида, диапазона или меры", () => {
  for (const [field, value] of [
    ["kind", "fine"],
    ["execution", "conditional"],
    ["suggested_value", 0],
    ["range_min", 1000],
    ["range_max", undefined],
    ["unit", "rub"],
    ["selected_outcomes", 1],
    ["alternatives", undefined],
    ["data_source", "unknown"],
    ["reasons", [false]],
    ["legal_status", "passed"]
  ]) {
    const response = validAnalysis();
    response.recommendation[field] = value;
    assert.throws(() => ui.validateAnalysisResponse(response), new RegExp(`recommendation\\.${field}`));
  }
});

test("AnalysisResponse: форма, диапазон и число наблюдений согласованы", () => {
  const response = validAnalysis();
  response.recommendation = {
    ...response.recommendation,
    status: "available", kind: "imprisonment", execution: "conditional",
    suggested_value: 36, range_min: 24, range_max: 48, unit: "months",
    matching_cases: 7, usable_outcomes: 5, selected_outcomes: 4,
    reasons: ["Условное лишение свободы чаще всего встречается среди сопоставимых исходов"],
    legal_status: "not_checked"
  };
  assert.equal(ui.validateAnalysisResponse(response), response);

  for (const [field, value] of [
    ["suggested_value", 50],
    ["range_min", -10],
    ["range_max", "48"],
    ["execution", null],
    ["unit", "rub"],
    ["matching_cases", 4],
    ["usable_outcomes", 3],
    ["selected_outcomes", 0]
  ]) {
    const invalid = { ...response, recommendation: { ...response.recommendation, [field]: value } };
    assert.throws(() => ui.validateAnalysisResponse(invalid), /recommendation\./);
  }

  response.recommendation = {
    ...response.recommendation, kind: "fine", execution: null, unit: "rub",
    suggested_value: 150000, range_min: 100000, range_max: 250000
  };
  assert.equal(ui.validateAnalysisResponse(response), response);
  response.recommendation.execution = "conditional";
  assert.throws(() => ui.validateAnalysisResponse(response), /recommendation\.execution/);
});

test("AnalysisResponse: равная поддержка описывает два разных варианта без основной рекомендации", () => {
  const response = validAnalysis();
  const prison = {
    kind: "imprisonment", execution: "unspecified", suggested_value: 36,
    range_min: 24, range_max: 48, unit: "months", selected_outcomes: 3
  };
  const fine = {
    kind: "fine", execution: null, suggested_value: 150000,
    range_min: 100000, range_max: 250000, unit: "rub", selected_outcomes: 3
  };
  response.recommendation = {
    ...response.recommendation, matching_cases: 7, usable_outcomes: 6,
    alternatives: [prison, fine], data_source: "synthetic"
  };
  assert.equal(ui.validateAnalysisResponse(response), response);
  for (const alternatives of [
    [prison],
    [prison, prison],
    [prison, { ...fine, selected_outcomes: 4 }],
    [prison, { ...fine, selected_outcomes: 2 }],
    [prison, { ...fine, range_min: 200000 }]
  ]) {
    const invalid = { ...response, recommendation: { ...response.recommendation, alternatives } };
    assert.throws(() => ui.validateAnalysisResponse(invalid), /recommendation\.alternatives/);
  }

  response.recommendation = { ...response.recommendation, ...prison, status: "available" };
  assert.throws(() => ui.validateAnalysisResponse(response), /recommendation\.alternatives/);
});

test("Анализ отправляет обстоятельства дела и параметры поиска", () => {
  const fields = new FormData();
  for (const [name, value] of [
    ["article_part", "3"],
    ["amount_rub", "150000"],
    ["episodes", "1"],
    ["special_order", "on"],
    ["art_69", "on"],
    ["mitigating", "раскаяние"],
    ["mitigating", "явка с повинной"],
    ["aggravating", "рецидив преступлений"],
    ["narrative", "  Один эпизод  "],
    ["top_k", "10"],
    ["prison_months", "72"],
    ["conditional", "on"]
  ]) fields.append(name, value);
  assert.deepEqual(ui.buildRequest(fields), {
    case: {
      article_part: "3", amount_rub: 150000, episodes: 1,
      special_order: true, pretrial_agreement: false, art_64: false, art_69: true,
      mitigating: ["раскаяние", "явка с повинной"], aggravating: ["рецидив преступлений"], narrative: "Один эпизод"
    },
    top_k: 10
  });
});

test("API: HTML 200 и invalid JSON не считаются успешным ответом", async () => {
  global.fetch = async () => mockResponse({ contentType: "text/html", text: "<html>not JSON</html>" });
  await assert.rejects(() => ui.apiRequest("/analyses", { method: "POST", body: {} }), /не в формате JSON/);

  global.fetch = async () => mockResponse({
    json: async () => { throw new SyntaxError("Unexpected token"); }
  });
  await assert.rejects(() => ui.apiRequest("/models"), /некорректный JSON/);
});

test("API: text/html 404 заменяется понятным сообщением без разметки", async () => {
  const html = "<html><body><h1>Not Found</h1><p>secret route</p></body></html>";
  global.fetch = async () => mockResponse({ status: 404, contentType: "text/html; charset=utf-8", text: html });
  await assert.rejects(
    () => ui.apiRequest("/models"),
    (error) => {
      assert.equal(error.status, 404);
      assert.match(error.message, /ошибку 404/);
      assert.ok(!error.message.includes("<html"));
      assert.ok(!error.message.includes("secret route"));
      return true;
    }
  );
});

test("API: AbortError при чтении JSON не поглощается", async () => {
  global.fetch = async () => mockResponse({
    json: async () => { throw new DOMException("Body aborted", "AbortError"); }
  });
  await assert.rejects(
    () => ui.apiRequest("/models", { timeout: 15000 }),
    /15 секунд/
  );
});

test("422: payload.errors имеет приоритет над общим detail", () => {
  const message = ui.getApiErrorMessage(422, {
    detail: "Request validation failed",
    errors: [
      { loc: ["body", "case", "narrative"], msg: "String should have at most 4000 characters", type: "string_too_long" },
      { loc: ["body", "top_k"], msg: "Input should be greater than 0", type: "greater_than" }
    ]
  });
  assert.match(message, /case → narrative/);
  assert.match(message, /4000/);
  assert.match(message, /top_k/);
  assert.doesNotMatch(message, /Request validation failed/);
});

test("reset формы восстанавливает сохранённый token", () => {
  const tokenInput = { value: "typed-token" };
  const article = { value: "4" };
  const targetForm = {
    reset() {
      tokenInput.value = "";
      article.value = "";
    }
  };
  const result = ui.resetFormPreservingToken(targetForm, tokenInput, "stored-token");
  assert.equal(result, "stored-token");
  assert.equal(tokenInput.value, "stored-token");
  assert.equal(article.value, "");
});

test("Сводка: count без единиц, q1/q3/iqr/median/min/max с единицами родителя", () => {
  assert.equal(ui.formatSummaryMetric(0, "prison_months", "count"), "0");
  assert.equal(ui.formatSummaryMetric(2, "fine_rub", "count"), "2");
  assert.equal(ui.formatSummaryMetric(3, "ban_years", "true_count"), "3");
  assert.equal(ui.formatSummaryMetric(4, "prison_months", "q1"), "4 мес.");
  assert.equal(ui.formatSummaryMetric(8, "prison_months", "q3"), "8 мес.");
  assert.equal(ui.formatSummaryMetric(4, "prison_months", "iqr"), "4 мес.");
  assert.equal(ui.formatSummaryMetric(12.5, "prison_months", "median"), "12,5 мес.");
  assert.equal(ui.formatSummaryMetric(2, "ban_years", "min"), "2 лет");
  assert.equal(ui.formatSummaryMetric(5, "ban_years", "max"), "5 лет");
  assert.match(ui.formatSummaryMetric(1000, "fine_rub", "median"), /₽$/);
  assert.match(ui.formatSummaryMetric(0.25, "conditional", "rate"), /^25\s*%$/);
  assert.doesNotMatch(ui.formatSummaryMetric(0.25, "conditional", "rate"), /мес\.|₽|лет/);
  assert.equal(ui.formatSummaryMetric(7, "returned_count"), "7");
});

test("Наказания: boolean fine_multiple и ban_on_positions отображаются по смыслу", () => {
  const tags = ui.getPunishmentTags({ fine_multiple: true, ban_on_positions: true, ban_years: null });
  assert.deepEqual(tags, ["кратный штраф", "запрет на занятие должностей"]);
  assert.ok(!tags.some((tag) => /true|false/.test(tag)));

  const withYears = ui.getPunishmentTags({ fine_multiple: false, ban_on_positions: true, ban_years: 2.5 });
  assert.deepEqual(withYears, ["запрет занимать должности на 2,5 лет"]);
  assert.equal(withYears.some((tag) => /кратн/.test(tag)), false);
});

test("POST-блокировка затрагивает все редактируемые элементы формы", () => {
  const controls = [{ disabled: false }, { disabled: false }, { disabled: false }];
  const targetForm = { querySelectorAll: () => controls };
  ui.setFormControlsDisabled(targetForm, true);
  assert.ok(controls.every((control) => control.disabled));
  ui.setFormControlsDisabled(targetForm, false);
  assert.ok(controls.every((control) => !control.disabled));
});

test("Уровни проверок доступны текстом, включая not_checked", () => {
  assert.equal(ui.severityLabel("error"), "Ошибка");
  assert.equal(ui.severityLabel("warning"), "Предупреждение");
  assert.equal(ui.severityLabel("passed"), "Проверка пройдена");
  assert.equal(ui.severityLabel("not_checked"), "Проверка не выполнена");
});

test("Форма соответствует API: ст. 290 части 1–6 и только данные дела", () => {
  const html = fs.readFileSync(path.join(__dirname, "../ui/index.html"), "utf8");
  const select = html.match(/<select id="article-part"[\s\S]*?<\/select>/)[0];
  const values = [...select.matchAll(/<option value="([1-6])">/g)].map((match) => match[1]);
  assert.deepEqual(values, ["1", "2", "3", "4", "5", "6"]);
  assert.doesNotMatch(html, /ст\. 158/);
  assert.match(html, /name="narrative"[^>]*maxlength="4000"/);
  assert.match(html, /name="amount_rub"[^>]*max="1000000000000"[^>]*step="0\.01"/);
  assert.match(html, /name="episodes"[^>]*max="1000"/);
  assert.doesNotMatch(html, /name="(?:prison_months|fine_rub|fine_multiple|ban_years|conditional|confiscation)"/);
  assert.ok(html.indexOf('id="recommendation-title"') > 0);
  assert.ok(html.indexOf('id="recommendation-title"') < html.indexOf('id="legal-title"'));
});

test("Мобильный API status остаётся видимым текстом", () => {
  const css = fs.readFileSync(path.join(__dirname, "../ui/styles.css"), "utf8");
  const mobileStatus = css.match(/@media \(max-width: 560px\)[\s\S]*?#api-status-text\s*\{([^}]*)\}/);
  assert.ok(mobileStatus, "Мобильное правило статуса не найдено");
  assert.match(mobileStatus[1], /white-space:\s*nowrap/);
  assert.doesNotMatch(mobileStatus[1], /display:\s*none|clip:\s*rect|width:\s*1px/);
});
