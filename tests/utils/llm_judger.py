import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


class LLMJudger:
    def __init__(self, api_keys_path: str = "tests/api_keys.json"):
        """
        Initialize the LLM Judger with API keys.
        Defaults to using OpenAI (gpt-4o) if keys are available, otherwise falls back to Qwen.
        """
        self.api_keys = self._load_api_keys(api_keys_path)
        self.llms = self._init_llms()
        self._results: List[Dict[str, Any]] = []

    def _load_api_keys(self, path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            logger.warning(f"API keys file not found at {path}. Judger might fail.")
            return {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load API keys: {e}")
            return {}

    def _init_llms(self):
        """
        Initialize the LLM Judger, collecting multiple API providers in order of preference.
        """
        providers = [
            {
                "name": "OpenAI",
                "key_name": "assistApiKeyOpenai",
                "model": "gpt-4o",
                "base_url": "https://api.openai.com/v1"
            },
            {
                "name": "SiliconFlow",
                "key_name": "assistApiKeySilicon",
                "model": "deepseek-ai/DeepSeek-V3",
                "base_url": "https://api.siliconflow.cn/v1"
            },
            {
                "name": "Qwen",
                "key_name": "assistApiKeyQwen",
                "model": "qwen-max",
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
            },
            {
                "name": "GLM",
                "key_name": "assistApiKeyGlm",
                "model": "glm-4-plus",
                "base_url": "https://open.bigmodel.cn/api/paas/v4/"
            }
        ]

        llms = []
        for p in providers:
            api_key = self.api_keys.get(p["key_name"])
            # Some example files use "sk-..." or similar placeholders
            if api_key and api_key != "sk-..." and not api_key.startswith("your_"):
                try:
                    llm = ChatOpenAI(
                        model=p["model"],
                        api_key=api_key,
                        base_url=p["base_url"],
                        max_retries=1,
                        request_timeout=30
                    )
                    llms.append({"llm": llm, "name": p["name"]})
                except Exception as e:
                    logger.warning(f"Failed to init {p['name']}: {e}")
                    
        if not llms:
            logger.warning("No valid API key found for LLM Judger. Auto-pass mode enabled.")
        return llms

    def judge(self, input_text: str, output_text: str, criteria: str,
              test_name: str = "") -> bool:
        """
        Evaluate if the output_text satisfies the criteria given the input_text.
        Records the result internally for report generation.
        Returns True if passed, False otherwise.
        """
        result_entry = {
            "timestamp": datetime.now().isoformat(),
            "test_name": test_name,
            "input": input_text[:1000],    # Increased truncation limit
            "output": output_text[:2000],  # Increased truncation limit
            "criteria": criteria,
            "passed": False,
            "error": None,
            "verdict": None
        }

        if not hasattr(self, 'llms'):
            self.llms = self._init_llms()

        if not self.llms:
            logger.warning("LLM Judger not initialized, skipping check.")
            result_entry["passed"] = True
            result_entry["error"] = "No LLM configured, auto-pass"
            self._results.append(result_entry)
            return True

        prompt = f"""
You are an impartial, strict, and highly capable judge evaluating an AI assistant's response.

[User Input]: {input_text}
[AI Response]: {output_text}

[Evaluation Criteria]: {criteria}

Carefully consider whether the AI Response satisfies all elements of the Evaluation Criteria based on the User Input.
Your final answer must be exactly one word: either "YES" or "NO". Do NOT provide any explanation or extra text.
        """
        
        last_error = None
        for provider_info in self.llms:
            llm = provider_info["llm"]
            provider_name = provider_info["name"]
            try:
                logger.info(f"Attempting judgement with {provider_name}...")
                response = llm.invoke([HumanMessage(content=prompt.strip())])
                verdict = response.content.strip().upper()
                
                # Clean up verdict just in case the model added punctuation like "YES." or "YES!"
                clean_verdict = verdict.replace(".", "").replace("!", "").replace("'", "").replace('"', "").strip()
                
                if clean_verdict.startswith("YES"):
                    passed = True
                elif clean_verdict.startswith("NO"):
                    passed = False
                else:
                    passed = "YES" in clean_verdict # Fallback
                    logger.warning(f"Unexpected LLM Judgement format from {provider_name}: '{verdict}'. Evaluated as passed={passed}.")
                    
                logger.info(f"Judgement [{test_name}] via {provider_name}: {clean_verdict} (Criteria: {criteria})")
                
                result_entry["verdict"] = verdict
                result_entry["passed"] = passed
                self._results.append(result_entry)
                return passed
            except Exception as e:
                logger.warning(f"LLM Judger failed with {provider_name}: {e}")
                last_error = e
                continue
                
        # If all providers failed
        logger.error(f"All LLM Judger providers failed. Last error: {last_error}")
        result_entry["error"] = str(last_error)
        self._results.append(result_entry)
        return False

    @property
    def results(self) -> List[Dict[str, Any]]:
        return self._results

    def generate_report(self, output_dir: str = "tests/reports") -> Optional[str]:
        """
        Generate a Markdown + JSON report of all judged results.
        Returns the path to the Markdown report, or None if no results.
        """
        if not self._results:
            logger.info("No LLM Judger results to report.")
            return None

        report_dir = Path(output_dir)
        report_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        md_path = report_dir / f"test_report_{ts}.md"
        json_path = report_dir / f"test_report_{ts}.json"

        total = len(self._results)
        passed = sum(1 for r in self._results if r["passed"])
        failed = total - passed

        # --- JSON report ---
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "generated_at": datetime.now().isoformat(),
                "summary": {"total": total, "passed": passed, "failed": failed},
                "results": self._results,
            }, f, ensure_ascii=False, indent=2)

        # --- Markdown report ---
        lines = [
            f"# N.E.K.O. Test Report — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            f"- **Total checks**: {total}",
            f"- **Passed**: {passed}",
            f"- **Failed**: {failed}",
            "",
            "## Details",
            "",
            "| # | Test | Input | Output (truncated) | Criteria | Result |",
            "|---|---|---|---|---|---|",
        ]

        for i, r in enumerate(self._results, 1):
            icon = "✅" if r["passed"] else "❌"
            inp = r["input"].replace("|", "\\|").replace("\n", " ")[:60]
            out = r["output"].replace("|", "\\|").replace("\n", " ")[:80]
            crit = r["criteria"].replace("|", "\\|")[:60]
            name = r["test_name"] or f"check_{i}"
            error_note = f" ⚠️ {r['error']}" if r.get("error") else ""
            lines.append(f"| {i} | {name} | {inp} | {out} | {crit} | {icon}{error_note} |")

        lines.append("")
        lines.append(f"_JSON data: [{json_path.name}]({json_path.name})_")
        lines.append("")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"\n{'='*60}")
        print(f"📋 Test Report: {md_path.resolve()}")
        print(f"   JSON Data:   {json_path.resolve()}")
        print(f"   Results:     {passed}/{total} passed")
        print(f"{'='*60}\n")

        return str(md_path)
