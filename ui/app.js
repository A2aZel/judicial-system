"use strict";

(() => {
  const API_BASE = "/api/v1";
  const TOKEN_KEY = "judicialAnalyticsBearerToken";
  const REQUEST_TIMEOUT_MS = 60000;

  const hasDOM = typeof document !== "undefined";
  const form = hasDOM ? document.getElementById("analysis-form") : null;
  const submitButton = hasDOM ? document.getElementById("submit-analysis") : null;
  const submitLabel = hasDOM ? document.getElementById("submit-label") : null;
  const submitArrow = hasDOM ? submitButton.querySelector(".submit-arrow") : null;
  const submitSpinner = hasDOM ? submitButton.querySelector(".button-spinner") : null;
  const refreshButton = hasDOM ? document.getElementById("refresh-status") : null;
  const tokenInput = hasDOM ? document.getElementById("api-token") : null;
  const tokenDetails = hasDOM ? document.getElementById("token-details") : null;
  const emptyState = hasDOM ? document.getElementById("empty-state") : null;
  const errorState = hasDOM ? document.getElementById("analysis-error") : null;
  const results = hasDOM ? document.getElementById("analysis-results") : null;
  const requestStatus = hasDOM ? document.getElementById("request-status") : null;
  const printButton = hasDOM ? document.getElementById("print-results") : null;
  const formErrors = hasDOM ? document.getElementById("form-errors") : null;

  let isSubmitting = false;
  let isCheckingService = false;
  let lastRequest = null;
  let currentAnalysis = null;

  const numberFormatter = new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 3 });
  const moneyFormatter = new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 2 });
  const percentFormatter = new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 1, style: "percent" });

  function createElement(tag, options = {}, children = []) {
    const node = document.createElement(tag);
    if (options.className) node.className = options.className;
    if (options.text !== undefined && options.text !== null) node.textContent = String(options.text);
    if (options.attributes) {
      Object.entries(options.attributes).forEach(([name, value]) => {
        if (value !== undefined && value !== null) node.setAttribute(name, String(value));
      });
    }
    const items = Array.isArray(children) ? children : [children];
    items.forEach((child) => {
      if (child === undefined || child === null) return;
      node.append(child instanceof Node ? child : document.createTextNode(String(child)));
    });
    return node;
  }

  function replaceContent(target, ...children) {
    target.replaceChildren(...children.filter(Boolean));
  }

  function isObject(value) {
    return value !== null && typeof value === "object" && !Array.isArray(value);
  }

  function asArray(value) {
    if (Array.isArray(value)) return value;
    if (value === undefined || value === null || value === "") return [];
    return [value];
  }

  function firstValue(object, keys) {
    if (!isObject(object)) return undefined;
    for (const key of keys) {
      if (object[key] !== undefined && object[key] !== null && object[key] !== "") return object[key];
    }
    return undefined;
  }

  function asFiniteNumber(value) {
    if (value === "" || value === null || value === undefined || typeof value === "boolean") return null;
    const number = typeof value === "number" ? value : Number(String(value).replace(/\s/g, "").replace(",", "."));
    return Number.isFinite(number) ? number : null;
  }

  function formatNumber(value, suffix = "") {
    const number = asFiniteNumber(value);
    return number === null ? String(value) : `${numberFormatter.format(number)}${suffix}`;
  }

  function formatMoney(value) {
    const number = asFiniteNumber(value);
    return number === null ? String(value) : `${moneyFormatter.format(number)} ₽`;
  }

  function formatBoolean(value) {
    if (value === true || value === "true" || value === 1) return "Да";
    if (value === false || value === "false" || value === 0) return "Нет";
    return String(value);
  }

  function humanizeKey(key) {
    const labels = {
      status: "Статус",
      state: "Состояние",
      name: "Название",
      model: "Модель",
      model_name: "Модель",
      version: "Версия",
      type: "Тип",
      task: "Задача",
      loaded: "Загружена",
      available: "Доступна",
      ready: "Готова",
      checkpoint: "Checkpoint",
      checkpoint_status: "Статус checkpoint",
      device: "Устройство",
      embedding_model: "Модель эмбеддингов",
      retrieval_model: "Модель поиска",
      risk_model: "Модель риска",
      appeal_model: "Модель риска апелляции",
      trained: "Обучена",
      verified: "Проверена",
      validation: "Валидация",
      updated_at: "Обновлено",
      count: "Количество",
      candidate_count: "Кандидатов в выборке",
      returned_count: "Возвращено аналогов",
      true_count: "Количество «да»",
      rate: "Доля «да»",
      total: "Всего",
      min: "Минимум",
      max: "Максимум",
      mean: "Среднее",
      median: "Медиана",
      q1: "Первый квартиль (Q1)",
      q3: "Третий квартиль (Q3)",
      iqr: "Межквартильный размах",
      p25: "25-й перцентиль",
      p75: "75-й перцентиль",
      prison_months: "Лишение свободы",
      fine_rub: "Штраф",
      fine_multiple: "Кратность штрафа",
      ban_years: "Ограничение деятельности",
      conditional: "Условное осуждение",
      confiscation: "Конфискация",
      distance: "Расстояние",
      score: "Оценка",
      code: "Код",
      source: "Источник",
      message: "Сообщение",
      detail: "Подробности",
      description: "Описание"
    };
    if (labels[key]) return labels[key];
    return String(key)
      .replace(/_/g, " ")
      .replace(/^./, (letter) => letter.toLocaleUpperCase("ru-RU"));
  }

  function formatScalar(value, key = "") {
    if (typeof value === "boolean") return formatBoolean(value);
    if (asFiniteNumber(value) !== null) {
      if (key === "fine_rub") return formatMoney(value);
      if (key === "prison_months") return formatNumber(value, " мес.");
      if (key === "ban_years") return formatNumber(value, " лет");
      return numberFormatter.format(asFiniteNumber(value));
    }
    return String(value);
  }

  function readToken() {
    try {
      return sessionStorage.getItem(TOKEN_KEY) || "";
    } catch (_error) {
      return "";
    }
  }

  function writeToken(value) {
    try {
      if (value) sessionStorage.setItem(TOKEN_KEY, value);
      else sessionStorage.removeItem(TOKEN_KEY);
    } catch (_error) {
      updateTokenState(false);
      return false;
    }
    updateTokenState(Boolean(value));
    return true;
  }

  function updateTokenState(isSet) {
    const tokenState = document.getElementById("token-state");
    tokenState.textContent = isSet ? "Токен сохранён" : "Токен не задан";
    tokenState.classList.toggle("is-set", isSet);
  }

  async function apiRequest(path, options = {}) {
    const controller = new AbortController();
    const timeoutMs = options.timeout || REQUEST_TIMEOUT_MS;
    const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
    const headers = new Headers(options.headers || {});
    headers.set("Accept", "application/json");
    const token = readToken();
    if (token) headers.set("Authorization", `Bearer ${token}`);
    if (options.body !== undefined) headers.set("Content-Type", "application/json");

    try {
      const response = await fetch(`${API_BASE}${path}`, {
        method: options.method || "GET",
        headers,
        body: options.body === undefined ? undefined : JSON.stringify(options.body),
        signal: controller.signal,
        credentials: "same-origin"
      });

      const contentType = (response.headers.get("content-type") || "").toLowerCase();
      const isJSON = contentType.includes("application/json") || contentType.includes("+json");
      let payload = null;
      if (response.status !== 204) {
        if (isJSON) {
          try {
            payload = await response.json();
          } catch (error) {
            if (error.name === "AbortError") throw error;
            throw new Error("Сервис вернул некорректный JSON. Ответ не принят.");
          }
        } else {
          const text = await response.text();
          if (response.ok) {
            throw new Error("Сервис вернул ответ не в формате JSON. Ответ не принят.");
          }
          const isHTML = contentType.includes("text/html") || /<\/?[a-z][^>]*>/i.test(text);
          if (text && !isHTML && contentType.startsWith("text/plain")) {
            payload = { detail: text.slice(0, 500) };
          }
        }
      }

      if (!response.ok) {
        const error = new Error(getApiErrorMessage(response.status, payload));
        error.status = response.status;
        throw error;
      }
      if (response.status === 204) {
        throw new Error("Сервис вернул пустой ответ вместо JSON. Ответ не принят.");
      }
      return payload;
    } catch (error) {
      if (error.name === "AbortError") {
        const timeoutError = new Error(`Сервис не ответил в течение ${Math.round(timeoutMs / 1000)} секунд. Проверьте его состояние и повторите запрос.`);
        timeoutError.cause = error;
        throw timeoutError;
      }
      if (error instanceof TypeError) {
        throw new Error("Сервис API недоступен. Проверьте, что backend запущен и интерфейс открыт через него.");
      }
      throw error;
    } finally {
      window.clearTimeout(timeout);
    }
  }

  function getApiErrorMessage(status, payload) {
    if (status === 401 || status === 403) {
      return "API отклонил запрос авторизации. Укажите действующий Bearer-токен в разделе «Доступ к API» и повторите попытку.";
    }
    if (status === 422) {
      const details = payload && (payload.errors || payload.detail || payload.message);
      if (Array.isArray(details) && details.length) {
        const messages = details.map((item) => {
          if (!isObject(item)) return String(item);
          const location = Array.isArray(item.loc) ? item.loc.filter((part) => part !== "body").join(" → ") : "";
          const message = firstValue(item, ["msg", "message"]) || "Ошибка в поле";
          return location ? `${location}: ${message}` : String(message);
        });
        return `Проверьте исходные данные: ${messages.join("; ")}`;
      }
      const fallback = typeof details === "string" && !/<\/?[a-z][^>]*>/i.test(details) ? details : "API не принял структуру запроса";
      return `Проверьте исходные данные: ${fallback}.`;
    }
    if (status === 429) return "Превышен лимит запросов. Подождите и повторите попытку.";
    if (status >= 500) return "Сервис анализа временно недоступен или вернул внутреннюю ошибку. Результаты не получены.";
    const detail = payload && (payload.detail || payload.message || payload.error);
    if (typeof detail === "string" && detail.trim() && !/<\/?[a-z][^>]*>/i.test(detail)) return detail;
    return `Сервис вернул ошибку ${status}. Получите состояние API и повторите запрос позже.`;
  }

  function invalidAnalysisResponse(path, expectation) {
    const error = new Error(`API вернул некорректный AnalysisResponse: ${path} — ${expectation}. Результаты не показаны.`);
    error.code = "INVALID_ANALYSIS_RESPONSE";
    return error;
  }

  function requireResponse(condition, path, expectation) {
    if (!condition) throw invalidAnalysisResponse(path, expectation);
  }

  function isResponseNumber(value) {
    return typeof value === "number" && Number.isFinite(value);
  }

  function validateNumericSummary(summary, path) {
    requireResponse(isObject(summary), path, "ожидался объект");
    requireResponse(Number.isInteger(summary.count) && summary.count >= 0, `${path}.count`, "ожидалось целое число ≥ 0");
    ["median", "q1", "q3", "iqr", "min", "max"].forEach((key) => {
      requireResponse(Object.hasOwn(summary, key), `${path}.${key}`, "поле обязательно");
      const value = summary[key];
      requireResponse(value === null || isResponseNumber(value), `${path}.${key}`, "ожидалось число или null");
    });
  }

  function validateBooleanSummary(summary, path) {
    requireResponse(isObject(summary), path, "ожидался объект");
    ["count", "true_count"].forEach((key) => {
      requireResponse(Number.isInteger(summary[key]) && summary[key] >= 0, `${path}.${key}`, "ожидалось целое число ≥ 0");
    });
    requireResponse(Object.hasOwn(summary, "rate"), `${path}.rate`, "поле обязательно");
    requireResponse(summary.rate === null || (isResponseNumber(summary.rate) && summary.rate >= 0 && summary.rate <= 1), `${path}.rate`, "ожидалось число от 0 до 1 или null");
  }

  function validateAnalysisResponse(payload) {
    requireResponse(isObject(payload), "response", "ожидался JSON-объект");
    requireResponse(typeof payload.analysis_id === "string" && payload.analysis_id.trim().length > 0, "analysis_id", "ожидалась непустая строка");
    requireResponse(Array.isArray(payload.legal_checks), "legal_checks", "ожидался массив");
    requireResponse(Array.isArray(payload.analogues), "analogues", "ожидался массив");
    requireResponse(Array.isArray(payload.warnings), "warnings", "ожидался массив");

    payload.legal_checks.forEach((check, index) => {
      const path = `legal_checks[${index}]`;
      requireResponse(isObject(check), path, "ожидался объект");
      requireResponse(typeof check.code === "string", `${path}.code`, "ожидалась строка");
      requireResponse(["passed", "warning", "error", "not_checked"].includes(check.status), `${path}.status`, "неизвестный статус");
      requireResponse(typeof check.message === "string", `${path}.message`, "ожидалась строка");
    });
    payload.analogues.forEach((analogue, index) => {
      const path = `analogues[${index}]`;
      requireResponse(isObject(analogue), path, "ожидался объект");
      requireResponse(analogue.distance === null || (isResponseNumber(analogue.distance) && analogue.distance >= 0), `${path}.distance`, "ожидалось число ≥ 0 или null");
      ["prison_months", "fine_rub", "ban_years"].forEach((key) => {
        requireResponse(analogue[key] === null || analogue[key] === undefined || isResponseNumber(analogue[key]), `${path}.${key}`, "ожидалось число или null");
      });
      ["fine_multiple", "ban_on_positions", "conditional", "confiscation"].forEach((key) => {
        requireResponse(analogue[key] === null || analogue[key] === undefined || typeof analogue[key] === "boolean", `${path}.${key}`, "ожидался boolean или null");
      });
    });
    payload.warnings.forEach((warning, index) => {
      requireResponse(typeof warning === "string", `warnings[${index}]`, "ожидалась строка");
    });

    const summary = payload.punishment_summary;
    requireResponse(isObject(summary), "punishment_summary", "ожидался объект");
    ["candidate_count", "returned_count"].forEach((key) => {
      requireResponse(Number.isInteger(summary[key]) && summary[key] >= 0, `punishment_summary.${key}`, "ожидалось целое число ≥ 0");
    });
    ["prison_months", "fine_rub", "ban_years"].forEach((key) => validateNumericSummary(summary[key], `punishment_summary.${key}`));
    ["conditional", "confiscation"].forEach((key) => validateBooleanSummary(summary[key], `punishment_summary.${key}`));

    const risk = payload.appeal_risk;
    requireResponse(isObject(risk), "appeal_risk", "ожидался объект");
    requireResponse(["available", "unavailable"].includes(risk.status), "appeal_risk.status", "ожидался available или unavailable");
    requireResponse(Array.isArray(risk.reasons) && risk.reasons.every((reason) => typeof reason === "string"), "appeal_risk.reasons", "ожидался массив строк");
    requireResponse(risk.model === null || typeof risk.model === "string", "appeal_risk.model", "ожидалась строка или null");
    if (risk.status === "available") {
      requireResponse(isResponseNumber(risk.score) && risk.score >= 0 && risk.score <= 1, "appeal_risk.score", "для available ожидалось число от 0 до 1");
    } else {
      requireResponse(risk.score === null || risk.score === undefined, "appeal_risk.score", "для unavailable оценка должна быть null");
    }

    const modelInfo = payload.model_info;
    requireResponse(isObject(modelInfo), "model_info", "ожидался объект");
    requireResponse(typeof modelInfo.engine_version === "string" && modelInfo.engine_version.trim().length > 0, "model_info.engine_version", "ожидалась непустая строка");
    requireResponse(isObject(modelInfo.knn), "model_info.knn", "ожидался объект");
    requireResponse(isObject(modelInfo.bert), "model_info.bert", "ожидался объект");
    return payload;
  }

  function setServiceStatus(state, text) {
    const status = document.getElementById("api-status");
    status.dataset.state = state;
    status.title = text;
    document.getElementById("api-status-text").textContent = text;
  }

  function setModelsStatus(state, text) {
    const badge = document.getElementById("models-status");
    badge.dataset.state = state;
    badge.textContent = text;
  }

  async function checkService() {
    if (isCheckingService) return;
    isCheckingService = true;
    refreshButton.disabled = true;
    setServiceStatus("loading", "Проверка API…");
    setModelsStatus("loading", "Загрузка");
    replaceContent(
      document.getElementById("models-content"),
      createElement("div", { className: "inline-loading" }, [
        createElement("span", { className: "spinner", attributes: { "aria-hidden": "true" } }),
        createElement("span", { text: "Запрос состояния и сведений о моделях…" })
      ])
    );

    const [healthResult, modelsResult] = await Promise.allSettled([
      apiRequest("/health", { timeout: 15000 }),
      apiRequest("/models", { timeout: 15000 })
    ]);

    const health = healthResult.status === "fulfilled" ? healthResult.value : null;
    const serviceState = isObject(health) ? firstValue(health, ["status", "state"]) : null;
    if (typeof serviceState === "string" && serviceState.trim()) {
      const stateText = serviceState.toLowerCase() !== "ok" ? `API отвечает: ${serviceState}` : "API доступен";
      setServiceStatus("online", stateText);
    } else {
      setServiceStatus("error", healthResult.status === "fulfilled" ? "Некорректный ответ API" : "API недоступен");
    }

    const modelsValid = modelsResult.status === "fulfilled" && (Array.isArray(modelsResult.value) || isObject(modelsResult.value));
    if (modelsValid) {
      renderModels(modelsResult.value);
      setModelsStatus("online", "Данные получены");
      document.getElementById("models-caption").textContent = "Сведения получены от GET /api/v1/models";
    } else {
      setModelsStatus("error", "Недоступно");
      document.getElementById("models-caption").textContent = "Сведения не получены; предположения не отображаются";
      const message = modelsResult.status === "rejected"
        ? modelsResult.reason.message || "Не удалось получить сведения о моделях."
        : "Сервис вернул некорректные сведения о моделях.";
      replaceContent(
        document.getElementById("models-content"),
        createElement("div", { className: "neutral-empty" }, [
          createElement("p", { text: message })
        ])
      );
      if (modelsResult.status === "rejected" && [401, 403].includes(modelsResult.reason.status)) tokenDetails.open = true;
    }

    isCheckingService = false;
    refreshButton.disabled = false;
  }

  function normalizeModels(payload) {
    if (Array.isArray(payload)) return payload;
    if (!isObject(payload)) return payload === null ? [] : [payload];
    const list = firstValue(payload, ["models", "items", "data"]);
    if (Array.isArray(list)) return list;
    const scalarEntries = Object.entries(payload).filter(([, value]) => !isObject(value) && !Array.isArray(value));
    const objectEntries = Object.entries(payload).filter(([, value]) => isObject(value) || Array.isArray(value));
    const models = objectEntries.map(([name, value]) => {
      if (Array.isArray(value)) return { name, items: value.join(", ") };
      return { name, ...value };
    });
    if (scalarEntries.length) models.unshift(Object.fromEntries(scalarEntries));
    return models;
  }

  function renderModels(payload) {
    const models = normalizeModels(payload);
    const container = document.getElementById("models-content");
    if (!models.length) {
      replaceContent(container, createNeutralEmpty("Сервис не вернул сведений о моделях."));
      return;
    }
    const cards = models.map((model, index) => createModelCard(model, index));
    replaceContent(container, createElement("div", { className: "model-list" }, cards));
  }

  function createModelCard(model, index) {
    if (!isObject(model)) {
      return createElement("div", { className: "model-card" }, [
        createElement("p", { className: "model-card-title", text: `Модель ${index + 1}` }),
        createElement("p", { className: "message-text", text: String(model) })
      ]);
    }
    const title = firstValue(model, ["name", "model_name", "model", "id", "task", "type"]) || `Сведения ${index + 1}`;
    const details = Object.entries(model).filter(([key, value]) => {
      return !["name", "model_name", "model", "id"].includes(key) && value !== null && value !== undefined && value !== "";
    });
    const card = createElement("div", { className: "model-card" }, [
      createElement("p", { className: "model-card-title", text: title })
    ]);
    if (details.length) {
      const list = createElement("dl", { className: "model-details" });
      details.forEach(([key, value]) => {
        list.append(
          createElement("dt", { text: humanizeKey(key) }),
          createElement("dd", { text: displayValue(value, key) })
        );
      });
      card.append(list);
    }
    return card;
  }

  function displayValue(value, key = "", depth = 0) {
    if (value === null || value === undefined || value === "") return "Не указано";
    if (typeof value === "boolean") return formatBoolean(value);
    if (typeof value !== "object") return formatScalar(value, key);
    if (depth > 1) {
      try { return JSON.stringify(value); } catch (_error) { return String(value); }
    }
    if (Array.isArray(value)) return value.map((item) => displayValue(item, key, depth + 1)).join("; ");
    return Object.entries(value)
      .map(([nestedKey, nestedValue]) => `${humanizeKey(nestedKey)}: ${displayValue(nestedValue, nestedKey, depth + 1)}`)
      .join("; ");
  }

  function getOptionalNumber(fieldName) {
    const field = form.elements[fieldName];
    if (!field || field.value.trim() === "") return undefined;
    return Number(field.value);
  }

  function buildRequest() {
    const data = new FormData(form);
    const caseData = {
      article_part: String(data.get("article_part") || "").trim(),
      special_order: data.has("special_order"),
      pretrial_agreement: data.has("pretrial_agreement"),
      art_64: data.has("art_64"),
      art_69: data.has("art_69"),
      mitigating: data.getAll("mitigating").map(String),
      aggravating: data.getAll("aggravating").map(String)
    };
    const amount = getOptionalNumber("amount_rub");
    const episodes = getOptionalNumber("episodes");
    const narrative = String(data.get("narrative") || "").trim();
    if (amount !== undefined) caseData.amount_rub = amount;
    if (episodes !== undefined) caseData.episodes = episodes;
    if (narrative) caseData.narrative = narrative;

    const draftPunishment = {
      conditional: data.has("conditional"),
      confiscation: data.has("confiscation")
    };
    ["prison_months", "fine_rub", "fine_multiple", "ban_years"].forEach((key) => {
      const value = getOptionalNumber(key);
      if (value !== undefined) draftPunishment[key] = value;
    });

    const payload = { case: caseData, draft_punishment: draftPunishment };
    const topK = getOptionalNumber("top_k");
    if (topK !== undefined) payload.top_k = topK;
    return payload;
  }

  function validateForm() {
    const errors = [];
    form.querySelectorAll("input, select, textarea").forEach((field) => field.removeAttribute("aria-invalid"));
    const article = form.elements.article_part;
    if (!article.value.trim()) {
      errors.push("Выберите часть статьи 290 УК РФ.");
      article.setAttribute("aria-invalid", "true");
    }
    form.querySelectorAll("input[type=number]").forEach((field) => {
      if (!field.validity.valid) {
        field.setAttribute("aria-invalid", "true");
        const label = form.querySelector(`label[for="${field.id}"]`);
        errors.push(`${label ? label.textContent.replace("необязательно", "").trim() : field.name}: проверьте числовое значение и допустимый диапазон.`);
      }
    });
    const fineMultiple = form.elements.fine_multiple;
    if (fineMultiple.value.trim() !== "" && Number(fineMultiple.value) <= 0) {
      fineMultiple.setAttribute("aria-invalid", "true");
      errors.push("Кратность штрафа должна быть больше нуля.");
    }
    if (errors.length) {
      replaceContent(
        formErrors,
        createElement("p", { text: "Исправьте данные перед отправкой:" }),
        createElement("ul", {}, errors.map((error) => createElement("li", { text: error })))
      );
      formErrors.hidden = false;
      formErrors.focus();
      const firstInvalid = form.querySelector("[aria-invalid=true]");
      if (firstInvalid) firstInvalid.focus({ preventScroll: false });
      return false;
    }
    formErrors.hidden = true;
    replaceContent(formErrors);
    return true;
  }

  function setFormControlsDisabled(targetForm, disabled) {
    targetForm.querySelectorAll("input, select, textarea, button").forEach((control) => {
      control.disabled = disabled;
    });
  }

  function setSubmitting(value) {
    isSubmitting = value;
    setFormControlsDisabled(form, value);
    document.getElementById("reset-form").disabled = value;
    refreshButton.disabled = value || isCheckingService;
    submitLabel.textContent = value ? "Выполняется анализ…" : "Выполнить анализ";
    submitSpinner.hidden = !value;
    submitArrow.hidden = value;
    submitButton.setAttribute("aria-busy", String(value));
    form.setAttribute("aria-busy", String(value));
  }

  async function submitAnalysis(payload) {
    if (isSubmitting || !validateForm()) return;
    let focusTokenAfterRequest = false;
    writeToken(tokenInput.value.trim());
    lastRequest = payload || buildRequest();
    currentAnalysis = null;
    clearAnalysisModelInfo();
    setSubmitting(true);
    errorState.hidden = true;
    emptyState.hidden = true;
    results.hidden = true;
    printButton.hidden = true;
    requestStatus.textContent = "Запрос отправлен. Форма заблокирована до получения ответа.";

    try {
      const response = await apiRequest("/analyses", { method: "POST", body: lastRequest });
      validateAnalysisResponse(response);
      currentAnalysis = response;
      renderAnalysis(response);
      emptyState.hidden = true;
      errorState.hidden = true;
      results.hidden = false;
      printButton.hidden = false;
      requestStatus.textContent = "Анализ завершён. Результаты доступны ниже.";
      results.focus({ preventScroll: true });
      results.scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
    } catch (error) {
      currentAnalysis = null;
      clearAnalysisModelInfo();
      results.hidden = true;
      emptyState.hidden = true;
      errorState.hidden = false;
      printButton.hidden = true;
      document.getElementById("analysis-error-message").textContent = error.message || "Не удалось получить результаты анализа.";
      requestStatus.textContent = `Анализ не выполнен. ${error.message || "Ошибка сервиса."}`;
      if ([401, 403].includes(error.status)) {
        tokenDetails.open = true;
        focusTokenAfterRequest = true;
      } else {
        errorState.scrollIntoView({ behavior: "auto", block: "nearest" });
      }
    } finally {
      setSubmitting(false);
      if (focusTokenAfterRequest) tokenInput.focus({ preventScroll: true });
    }
  }

  function renderAnalysis(data) {
    document.getElementById("analysis-id").textContent = data.analysis_id || "Не возвращён API";
    renderLegalChecks(data.legal_checks);
    renderAppealRisk(data.appeal_risk);
    renderSummary(data.punishment_summary, data.analogues);
    renderDistribution(data.analogues, data.punishment_summary);
    renderAnalogues(data.analogues, data.model_info);
    renderWarnings(data.warnings);
    renderAnalysisModelInfo(data.model_info);
  }

  function normalizeMessage(item) {
    if (!isObject(item)) return { text: String(item), level: "info" };
    const level = String(firstValue(item, ["severity", "level", "status", "type"]) || "info").toLowerCase();
    const title = firstValue(item, ["title", "code", "rule", "check"]);
    const text = firstValue(item, ["message", "detail", "description", "text", "reason"]);
    const knownKeys = ["severity", "level", "status", "type", "title", "code", "rule", "check", "message", "detail", "description", "text", "reason"];
    const remaining = Object.entries(item).filter(([key]) => !knownKeys.includes(key));
    return {
      level,
      title: title ? String(title) : "",
      text: text !== undefined ? displayValue(text) : (remaining.length ? displayValue(item) : "Проверка выполнена"),
      meta: firstValue(item, ["source", "reference", "article"]),
      passed: item.passed
    };
  }

  function normalizeLevel(level, passed) {
    if (passed === true) return "success";
    if (passed === false) return "warning";
    if (["error", "critical", "fatal", "danger", "failed", "violation"].includes(level)) return "error";
    if (["warning", "warn", "attention"].includes(level)) return "warning";
    if (["success", "ok", "passed", "valid"].includes(level)) return "success";
    return "info";
  }

  function severityLabel(level, passed) {
    if (level === "not_checked") return "Проверка не выполнена";
    const normalized = normalizeLevel(level, passed);
    if (normalized === "error") return "Ошибка";
    if (normalized === "warning") return "Предупреждение";
    if (normalized === "success") return "Проверка пройдена";
    return "Информация";
  }

  function messageSymbol(level) {
    if (level === "error") return "!";
    if (level === "warning") return "!";
    if (level === "success") return "✓";
    return "i";
  }

  function createMessageItem(item) {
    const normalized = normalizeMessage(item);
    const level = normalizeLevel(normalized.level, normalized.passed);
    const body = createElement("div", { className: "message-body" });
    body.append(createElement("span", { className: "sr-only", text: `Уровень: ${severityLabel(normalized.level, normalized.passed)}. ` }));
    if (normalized.title) body.append(createElement("p", { className: "message-title", text: normalized.title }));
    body.append(createElement("p", { className: "message-text", text: normalized.text }));
    if (normalized.meta) body.append(createElement("span", { className: "message-meta", text: displayValue(normalized.meta) }));
    return createElement("li", { className: "message-item", attributes: { "data-level": level } }, [
      createElement("span", { className: "message-symbol", text: messageSymbol(level), attributes: { "aria-hidden": "true" } }),
      body
    ]);
  }

  function renderMessageList(target, items, emptyText) {
    const list = asArray(items);
    if (!list.length) {
      replaceContent(target, createNeutralEmpty(emptyText));
      return;
    }
    replaceContent(target, createElement("ul", { className: "message-list" }, list.map(createMessageItem)));
  }

  function renderLegalChecks(checks) {
    const list = asArray(checks);
    document.getElementById("legal-count").textContent = String(list.length);
    renderMessageList(document.getElementById("legal-checks"), list, "API не вернул отдельных юридических проверок.");
  }

  function renderWarnings(warnings) {
    const list = asArray(warnings);
    document.getElementById("warnings-count").textContent = String(list.length);
    renderMessageList(document.getElementById("analysis-warnings"), list, "Предупреждений нет.");
  }

  function renderAppealRisk(risk) {
    const container = document.getElementById("appeal-risk");
    const data = isObject(risk) ? risk : {};
    const status = String(firstValue(data, ["status", "state"]) || "unavailable").toLowerCase();
    const score = asFiniteNumber(firstValue(data, ["score", "probability", "value"]));
    const unavailableStatuses = ["unavailable", "not_available", "not-available", "disabled", "not_trained", "untrained", "no_checkpoint", "missing_checkpoint", "unknown", "none", "null"];
    const unavailable = unavailableStatuses.includes(status) || score === null;

    if (unavailable) {
      const block = createElement("div", { className: "risk-unavailable" }, [
        createElement("div", { className: "risk-value", text: "Недоступен до обучения проверенного checkpoint" }),
        createElement("p", { text: "Числовая оценка и процент не рассчитываются. API не предоставил применимый результат модели." })
      ]);
      replaceContent(container, block);
    } else {
      const scoreText = formatRiskScore(score);
      const children = [
        createElement("span", { className: "risk-status", text: humanizeRiskStatus(status) }),
        createElement("div", { className: "risk-value", text: scoreText })
      ];
      replaceContent(container, ...children);
    }

    const reasons = asArray(firstValue(data, ["reasons", "reason", "explanation"])).filter((item) => item !== null && item !== "");
    if (reasons.length) {
      container.append(createElement("ul", { className: "risk-reasons" }, reasons.map((reason) => createElement("li", { text: displayValue(reason) }))));
    }
    const model = firstValue(data, ["model", "model_info", "checkpoint"]);
    if (model) container.append(createElement("p", { className: "risk-model", text: `Модель: ${displayValue(model)}` }));
  }

  function formatRiskScore(score) {
    if (score >= 0 && score <= 1) return percentFormatter.format(score);
    if (score > 1 && score <= 100) return `${numberFormatter.format(score)} %`;
    return numberFormatter.format(score);
  }

  function humanizeRiskStatus(status) {
    const labels = {
      available: "Оценка модели",
      ok: "Оценка модели",
      ready: "Оценка модели",
      low: "Низкий по оценке модели",
      medium: "Средний по оценке модели",
      moderate: "Средний по оценке модели",
      high: "Высокий по оценке модели"
    };
    return labels[status] || `Статус: ${status}`;
  }

  function getPunishment(analogue) {
    if (!isObject(analogue)) return {};
    const nested = firstValue(analogue, ["punishment", "sentence", "draft_punishment"]);
    return { ...analogue, ...(isObject(nested) ? nested : {}) };
  }

  function getPrisonMonths(analogue) {
    const punishment = getPunishment(analogue);
    return asFiniteNumber(firstValue(punishment, ["prison_months", "imprisonment_months", "term_months"]));
  }

  function isConditional(analogue) {
    const punishment = getPunishment(analogue);
    const value = firstValue(punishment, ["conditional", "is_conditional", "suspended"]);
    return value === true || value === "true" || value === 1;
  }

  function renderSummary(summary, analogues) {
    const container = document.getElementById("punishment-summary");
    const list = createElement("dl", { className: "summary-list" });
    const rows = [];

    if (isObject(summary)) {
      Object.entries(summary).forEach(([key, value]) => {
        if (value === null || value === undefined || value === "") return;
        if (["distribution", "histogram", "buckets"].includes(key)) return;
        if (isObject(value)) {
          Object.entries(value).forEach(([nestedKey, nestedValue]) => {
            if (nestedValue === null || nestedValue === undefined || isObject(nestedValue) || Array.isArray(nestedValue)) return;
            rows.push([`${humanizeKey(key)} · ${humanizeKey(nestedKey)}`, formatSummaryMetric(nestedValue, key, nestedKey)]);
          });
        } else if (!Array.isArray(value)) {
          rows.push([humanizeKey(key), formatSummaryMetric(value, key)]);
        }
      });
    }

    if (!rows.length) {
      const listOfAnalogues = asArray(analogues);
      const prisonTerms = listOfAnalogues.map(getPrisonMonths).filter((value) => value !== null).sort((a, b) => a - b);
      rows.push(["Аналогов в выборке", numberFormatter.format(listOfAnalogues.length)]);
      if (prisonTerms.length) {
        rows.push(["Срок: минимум", formatNumber(prisonTerms[0], " мес.")]);
        rows.push(["Срок: медиана", formatNumber(median(prisonTerms), " мес.")]);
        rows.push(["Срок: максимум", formatNumber(prisonTerms[prisonTerms.length - 1], " мес.")]);
        rows.push(["Условное осуждение", `${listOfAnalogues.filter(isConditional).length} из ${listOfAnalogues.length}`]);
      }
    }

    if (!rows.length) {
      replaceContent(container, createNeutralEmpty("API не вернул данных для сводки."));
      return;
    }
    rows.forEach(([label, value]) => {
      list.append(createElement("div", { className: "summary-row" }, [
        createElement("dt", { text: label }),
        createElement("dd", { text: value })
      ]));
    });
    replaceContent(container, list);
  }

  function formatSummaryMetric(value, key, metric = "") {
    const statistic = metric || key;
    if (["count", "true_count", "candidate_count", "returned_count"].includes(statistic)) {
      return formatNumber(value);
    }
    if (statistic === "rate") {
      const rate = asFiniteNumber(value);
      return rate === null ? String(value) : percentFormatter.format(rate);
    }
    if (["fine_rub", "fine", "fine_amount"].some((part) => key.includes(part))) return formatMoney(value);
    if (["prison", "months", "term"].some((part) => key.includes(part))) return formatNumber(value, " мес.");
    if (["ban", "years"].some((part) => key.includes(part))) return formatNumber(value, " лет");
    return formatScalar(value, key);
  }

  function median(sortedValues) {
    const middle = Math.floor(sortedValues.length / 2);
    if (sortedValues.length % 2) return sortedValues[middle];
    return (sortedValues[middle - 1] + sortedValues[middle]) / 2;
  }

  function getDistribution(analogues, summary) {
    if (isObject(summary)) {
      const supplied = firstValue(summary, ["distribution", "histogram", "buckets"]);
      if (Array.isArray(supplied) && supplied.length) {
        return supplied.map((bucket, index) => {
          if (!isObject(bucket)) return { label: `Группа ${index + 1}`, count: asFiniteNumber(bucket) || 0 };
          return {
            label: String(firstValue(bucket, ["label", "name", "range", "bucket"]) || `Группа ${index + 1}`),
            count: asFiniteNumber(firstValue(bucket, ["count", "value", "total"])) || 0
          };
        });
      }
    }

    const list = asArray(analogues);
    const buckets = [
      { label: "До 1 года", count: 0, matches: (months) => months < 12 },
      { label: "1–3 года", count: 0, matches: (months) => months >= 12 && months < 36 },
      { label: "3–5 лет", count: 0, matches: (months) => months >= 36 && months < 60 },
      { label: "5–10 лет", count: 0, matches: (months) => months >= 60 && months < 120 },
      { label: "Свыше 10 лет", count: 0, matches: (months) => months >= 120 }
    ];
    let conditionalWithoutTerm = 0;
    let withoutTerm = 0;
    list.forEach((analogue) => {
      const months = getPrisonMonths(analogue);
      if (months === null) {
        if (isConditional(analogue)) conditionalWithoutTerm += 1;
        else withoutTerm += 1;
        return;
      }
      const bucket = buckets.find((item) => item.matches(months));
      if (bucket) bucket.count += 1;
    });
    const result = buckets.map(({ label, count }) => ({ label, count }));
    if (conditionalWithoutTerm) result.push({ label: "Условное, срок не указан", count: conditionalWithoutTerm });
    if (withoutTerm) result.push({ label: "Срок не указан", count: withoutTerm });
    return result;
  }

  function renderDistribution(analogues, summary) {
    const container = document.getElementById("punishment-distribution");
    const distribution = getDistribution(analogues, summary);
    const total = distribution.reduce((sum, bucket) => sum + bucket.count, 0);
    if (!distribution.length || total <= 0) {
      replaceContent(container, createNeutralEmpty("Недостаточно данных для построения распределения."));
      return;
    }
    const max = Math.max(...distribution.map((bucket) => bucket.count), 1);
    const chart = createElement("div", { className: "distribution-chart", attributes: { role: "list", "aria-label": `Распределение наказаний, всего наблюдений: ${numberFormatter.format(total)}` } });
    distribution.forEach((bucket) => {
      const share = bucket.count / total;
      const accessibleText = `${bucket.label}: ${numberFormatter.format(bucket.count)}, ${percentFormatter.format(share)}`;
      chart.append(createElement("div", { className: "distribution-row", attributes: { role: "listitem", "aria-label": accessibleText } }, [
        createElement("span", { className: "distribution-label", text: bucket.label, attributes: { "aria-hidden": "true" } }),
        createElement("div", { className: "distribution-track", attributes: { "aria-hidden": "true" } }, [
          createElement("div", { className: "distribution-bar", attributes: { style: `width: ${(bucket.count / max) * 100}%` } })
        ]),
        createElement("span", { className: "distribution-value", text: `${numberFormatter.format(bucket.count)} · ${percentFormatter.format(share)}`, attributes: { "aria-hidden": "true" } })
      ]));
    });
    const note = createElement("p", { className: "distribution-note", text: "Доля рассчитана по возвращённым API наблюдениям с ненулевым числом аналогов. Это описание выборки, а не прогноз исхода дела." });
    replaceContent(container, chart, note);
  }

  function renderAnalogues(analogues, modelInfo) {
    const list = asArray(analogues);
    document.getElementById("analogues-count").textContent = String(list.length);
    const rankingAvailable = modelInfo?.knn?.ranking_available === true;
    document.getElementById("analogues-description").textContent = rankingAvailable
      ? "Ранжирование по близости к введённым обстоятельствам"
      : "Неранжированная подвыборка cohort; порядок не означает близость дел";
    const container = document.getElementById("analogues-content");
    if (!list.length) {
      replaceContent(container, createNeutralEmpty("API не вернул судебных аналогов по заданным параметрам."));
      return;
    }

    const table = createElement("table", { className: "analogues-table" });
    table.append(createElement("caption", { className: "sr-only", text: "Судебные аналоги, расстояние и назначенное наказание" }));
    const head = createElement("thead");
    head.append(createElement("tr", {}, [
      createElement("th", { text: "Судебный акт и состав", attributes: { scope: "col" } }),
      createElement("th", { text: "Расстояние", attributes: { scope: "col" } }),
      createElement("th", { text: "Наказание по аналогу", attributes: { scope: "col" } })
    ]));
    const body = createElement("tbody");
    list.forEach((analogue, index) => body.append(createAnalogueRow(analogue, index)));
    table.append(head, body);
    replaceContent(container, createElement("div", {
      className: "table-wrap",
      attributes: { tabindex: "0", role: "region", "aria-label": "Таблица судебных аналогов с горизонтальной прокруткой" }
    }, [table]));
  }

  function createAnalogueRow(analogue, index) {
    const data = isObject(analogue) ? analogue : { text: String(analogue) };
    const title = firstValue(data, ["case_number", "title", "document_title", "name", "article_part", "article"]) || `Судебный акт ${index + 1}`;
    const court = firstValue(data, ["court", "court_name", "instance"]);
    const date = firstValue(data, ["date", "decision_date", "document_date"]);
    const id = firstValue(data, ["document_id", "doc_id", "case_id", "id", "url"]);
    const article = firstValue(data, ["article_part", "article", "qualification"]);
    const distance = firstValue(data, ["distance", "similarity_distance", "score_distance", "score"]);

    const titleCell = createElement("td", {}, [
      createElement("span", { className: "analogue-title", text: title })
    ]);
    const subtitle = [court, date, article && article !== title ? article : null].filter(Boolean).join(" · ");
    if (subtitle) titleCell.append(createElement("span", { className: "analogue-subtitle", text: subtitle }));
    if (id && String(id) !== String(title)) titleCell.append(createElement("span", { className: "analogue-id", text: id }));
    if (!isObject(analogue)) titleCell.append(createElement("span", { className: "analogue-subtitle", text: String(analogue) }));

    const distanceCell = createElement("td");
    if (distance !== undefined && distance !== null && distance !== "") {
      distanceCell.append(createElement("span", { className: "distance-value", text: formatNumber(distance) }));
      distanceCell.append(createElement("span", { className: "distance-hint", text: "меньше — ближе" }));
    } else {
      distanceCell.textContent = "Ранжирование недоступно";
    }

    const punishmentCell = createElement("td");
    const tags = getPunishmentTags(data);
    if (tags.length) {
      punishmentCell.append(createElement("div", { className: "punishment-tags" }, tags.map((tag, tagIndex) => createElement("span", {
        className: `punishment-tag${tagIndex === 0 ? " primary-tag" : ""}`,
        text: tag
      }))));
    } else {
      punishmentCell.textContent = firstValue(data, ["punishment_text", "sentence_text", "text"]) || "Наказание не указано";
    }

    return createElement("tr", {}, [titleCell, distanceCell, punishmentCell]);
  }

  function getPunishmentTags(analogue) {
    const punishment = getPunishment(analogue);
    const tags = [];
    const prison = firstValue(punishment, ["prison_months", "imprisonment_months", "term_months"]);
    const fine = firstValue(punishment, ["fine_rub", "fine", "fine_amount"]);
    const fineMultiple = firstValue(punishment, ["fine_multiple"]);
    const ban = firstValue(punishment, ["ban_years", "restriction_years"]);
    const banOnPositions = firstValue(punishment, ["ban_on_positions"]);
    const conditional = firstValue(punishment, ["conditional", "is_conditional", "suspended"]);
    const confiscation = firstValue(punishment, ["confiscation", "has_confiscation"]);

    if (prison !== undefined) tags.push(`${formatNumber(prison, " мес.")} лишения свободы`);
    if (conditional === true || conditional === "true" || conditional === 1) tags.push("условное");
    if (fine !== undefined) tags.push(`штраф ${formatMoney(fine)}`);
    if (fineMultiple === true) tags.push("кратный штраф");
    else if (fineMultiple !== undefined && fineMultiple !== null && typeof fineMultiple !== "boolean") tags.push(`кратность штрафа ${formatNumber(fineMultiple)}`);
    if (banOnPositions === true) {
      tags.push(ban !== undefined ? `запрет занимать должности на ${formatNumber(ban, " лет")}` : "запрет на занятие должностей");
    } else if (ban !== undefined) {
      tags.push(`ограничение деятельности на ${formatNumber(ban, " лет")}`);
    }
    if (confiscation === true || confiscation === "true" || confiscation === 1) tags.push("конфискация");
    return tags;
  }

  function clearAnalysisModelInfo() {
    const section = document.getElementById("analysis-model-info");
    section.hidden = true;
    replaceContent(document.getElementById("analysis-model-content"));
  }

  function renderAnalysisModelInfo(modelInfo) {
    const section = document.getElementById("analysis-model-info");
    const entries = isObject(modelInfo) ? Object.entries(modelInfo) : [];
    if (!entries.length) {
      clearAnalysisModelInfo();
      return;
    }
    section.hidden = false;
    renderModelsInto(document.getElementById("analysis-model-content"), modelInfo);
  }

  function renderModelsInto(container, payload) {
    const models = normalizeModels(payload);
    if (!models.length) {
      replaceContent(container, createNeutralEmpty("Сведения о модели анализа не возвращены."));
      return;
    }
    replaceContent(container, createElement("div", { className: "model-list" }, models.map((model, index) => createModelCard(model, index))));
  }

  function createNeutralEmpty(text) {
    return createElement("div", { className: "neutral-empty" }, [createElement("p", { text })]);
  }

  function resetFormPreservingToken(targetForm, input, storedToken) {
    targetForm.reset();
    input.value = storedToken;
    return storedToken;
  }

  if (hasDOM) {
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      submitAnalysis();
    });

    form.addEventListener("input", (event) => {
      if (event.target.matches("input, select, textarea")) {
        event.target.removeAttribute("aria-invalid");
        if (!form.querySelector("[aria-invalid=true]")) formErrors.hidden = true;
      }
    });

    document.getElementById("retry-analysis").addEventListener("click", () => {
      submitAnalysis(buildRequest());
    });

    document.getElementById("reset-form").addEventListener("click", () => {
      if (isSubmitting) return;
      resetFormPreservingToken(form, tokenInput, readToken());
      updateTokenState(Boolean(tokenInput.value));
      formErrors.hidden = true;
      form.querySelectorAll("[aria-invalid]").forEach((field) => field.removeAttribute("aria-invalid"));
      document.getElementById("narrative-counter").textContent = "0 / 4 000";
      document.getElementById("aggravating-count").hidden = true;
      results.hidden = true;
      errorState.hidden = true;
      emptyState.hidden = false;
      printButton.hidden = true;
      currentAnalysis = null;
      lastRequest = null;
      clearAnalysisModelInfo();
      requestStatus.textContent = "Форма и результаты очищены. Токен доступа сохранён.";
      form.elements.article_part.focus();
    });

    document.getElementById("narrative").addEventListener("input", (event) => {
      document.getElementById("narrative-counter").textContent = `${numberFormatter.format(event.target.value.length)} / 4 000`;
    });

    form.querySelectorAll("input[name=aggravating]").forEach((checkbox) => {
      checkbox.addEventListener("change", () => {
        const count = form.querySelectorAll("input[name=aggravating]:checked").length;
        const badge = document.getElementById("aggravating-count");
        badge.textContent = String(count);
        badge.hidden = count === 0;
      });
    });

    tokenInput.addEventListener("change", () => {
      writeToken(tokenInput.value.trim());
    });

    tokenInput.addEventListener("blur", () => {
      if (tokenInput.value.trim() !== readToken()) writeToken(tokenInput.value.trim());
    });

    document.getElementById("toggle-token").addEventListener("click", () => {
      const show = tokenInput.type === "password";
      tokenInput.type = show ? "text" : "password";
      const button = document.getElementById("toggle-token");
      button.setAttribute("aria-pressed", String(show));
      button.setAttribute("aria-label", show ? "Скрыть токен" : "Показать токен");
    });

    document.getElementById("clear-token").addEventListener("click", () => {
      tokenInput.value = "";
      writeToken("");
      requestStatus.textContent = "Токен удалён из sessionStorage.";
      tokenInput.focus();
    });

    refreshButton.addEventListener("click", checkService);
    printButton.addEventListener("click", () => {
      if (currentAnalysis) window.print();
    });

    tokenInput.value = readToken();
    updateTokenState(Boolean(tokenInput.value));
    checkService();
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = {
      apiRequest,
      validateAnalysisResponse,
      getApiErrorMessage,
      formatSummaryMetric,
      getPunishmentTags,
      resetFormPreservingToken,
      setFormControlsDisabled,
      severityLabel
    };
  }
})();
