"""Verify a running API using only the Python standard library."""

import argparse
import json
import math
import os
import sys
import urllib.error
import urllib.request


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def check(
    base_url: str, timeout: float, api_key: str | None, require_recommendation: bool = False
) -> tuple[int, int, str]:
    def fetch(path: str, payload: dict | None = None) -> tuple[bytes, str]:
        headers = {"Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        data = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            base_url.rstrip("/") + path, data=data, headers=headers
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read(), response.headers.get_content_type()

    def fetch_json(path: str, payload: dict | None = None) -> dict:
        content, content_type = fetch(path, payload)
        require(content_type == "application/json", f"{path}: expected JSON")
        result = json.loads(content)
        require(isinstance(result, dict), f"{path}: expected a JSON object")
        return result

    health = fetch_json("/api/v1/health")
    require(health.get("status") == "ok", "health: service is not ready")
    records = health.get("corpus", {}).get("records", 0)
    require(isinstance(records, int) and records > 0, "health: corpus is empty")

    models = fetch_json("/api/v1/models")
    require(
        models.get("knn", {}).get("algorithm") == "deterministic_knn",
        "models: structured kNN is not configured",
    )
    require(
        models.get("bert", {}).get("risk_enabled") is False,
        "models: appeal risk must be disabled for this profile",
    )
    ui, content_type = fetch("/")
    require(content_type == "text/html" and b"<html" in ui.lower(), "UI: HTML unavailable")

    analysis = fetch_json(
        "/api/v1/analyses",
        {
            "case": {
                "article_part": "3",
                "amount_rub": 150000,
                "episodes": 1,
                "special_order": False,
                "pretrial_agreement": False,
                "art_64": False,
                "art_69": False,
                "mitigating": [],
                "aggravating": [],
            },
            "top_k": 3,
        },
    )
    analogues = analysis.get("analogues")
    require(isinstance(analogues, list) and bool(analogues), "analysis: no analogues")
    require(
        analysis.get("model_info", {}).get("knn", {}).get("ranking_available") is True,
        "analysis: kNN ranking is unavailable",
    )
    distances = []
    for analogue in analogues:
        require(isinstance(analogue, dict), "analysis: invalid analogue")
        distance = analogue.get("distance")
        require(
            isinstance(distance, (int, float))
            and not isinstance(distance, bool)
            and math.isfinite(distance)
            and distance >= 0,
            "analysis: distance must be finite and nonnegative",
        )
        distances.append(distance)
    require(distances == sorted(distances), "analysis: analogues are not ranked")
    require(
        analysis.get("punishment_summary", {}).get("returned_count") == len(analogues),
        "analysis: returned count does not match analogues",
    )
    require(
        analysis.get("appeal_risk", {}).get("status") == "unavailable"
        and analysis.get("appeal_risk", {}).get("score") is None,
        "analysis: unexpected appeal risk result",
    )
    legal_checks = analysis.get("legal_checks", [])
    require(
        not any(check.get("status") == "error" for check in legal_checks),
        "analysis: facts-only request failed draft validation",
    )
    recommendation = analysis.get("recommendation", {})
    status = recommendation.get("status")
    require(status in {"available", "unavailable"}, "analysis: invalid recommendation status")
    require(
        recommendation.get("legal_status") == "not_checked" and bool(recommendation.get("reasons")),
        "analysis: invalid recommendation scope or explanation",
    )
    require(
        recommendation.get("data_source") in {"synthetic", "case_corpus"},
        "analysis: invalid recommendation data source",
    )
    require(not require_recommendation or status == "available", "analysis: no recommendation")
    values = [recommendation.get(key) for key in ("range_min", "suggested_value", "range_max")]
    if status == "available":
        require(
            all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                and value > 0
                for value in values
            ),
            "analysis: invalid recommendation values",
        )
        require(values == sorted(values), "analysis: recommendation lies outside its range")
    else:
        require(all(value is None for value in values), "analysis: unavailable recommendation has values")
    return records, len(analogues), status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument("--api-key-env", default="JUDICIAL_AI_API_KEY")
    parser.add_argument(
        "--require-recommendation", action="store_true", help="require a supported outcome"
    )
    args = parser.parse_args()
    if args.timeout <= 0 or not math.isfinite(args.timeout):
        parser.error("--timeout must be finite and greater than zero")
    try:
        records, analogues, recommendation = check(
            args.base_url,
            args.timeout,
            os.environ.get(args.api_key_env),
            args.require_recommendation,
        )
    except urllib.error.HTTPError as exc:
        print(f"Smoke failed: HTTP {exc.code}", file=sys.stderr)
        return 1
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"Smoke failed: connection error ({type(exc).__name__})", file=sys.stderr)
        return 1
    except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
        message = str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__
        print(f"Smoke failed: {message}", file=sys.stderr)
        return 1
    print(
        f"Smoke passed: records={records}, analogues={analogues}, ranked=true, "
        f"risk=unavailable, recommendation={recommendation}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
