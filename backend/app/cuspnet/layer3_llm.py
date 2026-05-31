import json
import re
from typing import Dict, List, Optional, Any


DISTORTION_TYPES = [
    "catastrophizing",
    "overgeneralization",
    "all_or_nothing",
    "emotional_reasoning",
    "personalization",
]

PRIMARY_DEFAULT = {
    "threat_identified": False,
    "threat_type": "unknown",
    "threat_intensity": 5,
    "threat_narrative": "",
    "primary_appraisal_score": 5,
}

SECONDARY_DEFAULT = {
    "coping_resources": [],
    "coping_efficacy": 5,
    "social_support_perceived": 5,
    "resource_narrative": "",
    "secondary_appraisal_score": 5,
}

DISTORTION_DEFAULT = []


class LazarusAppraisalChain:
    def __init__(
        self,
        model_name: str = r"D:\Models\huggingface\Qwen3.5-2B",
        device: str = "cuda",
        load_in_4bit: bool = True,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
    ):
        self.model_name = model_name
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens
        self.temperature = 0.1
        self.tokenizer = None
        self.model = None
        self._warmed_up = False

    def _load_model(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        self.model.eval()

    def _ensure_model(self):
        if self.model is None or self.tokenizer is None:
            self._load_model()
        if not self._warmed_up:
            self._warmup()

    def _warmup(self):
        try:
            import torch
            dummy_input = self.tokenizer("warmup", return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                _ = self.model.generate(
                    **dummy_input,
                    max_new_tokens=1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            self._warmed_up = True
        except Exception:
            self._warmed_up = True

    def _generate(self, prompt: str) -> str:
        self._ensure_model()
        import torch

        try:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            input_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature if self.temperature > 0 else 1e-5,
                    do_sample=self.temperature > 0,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            generated_ids = outputs[0][input_len:]
            raw = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            return raw
        except UnicodeDecodeError:
            try:
                raw_bytes = self.tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                )
                return raw_bytes.encode("utf-8", errors="replace").decode("utf-8")
            except Exception:
                return ""
        except Exception:
            return ""

    def _parse_json(self, text: str, default: dict = None) -> dict:
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                candidate = text[start:end]
                candidate = re.sub(r",\s*}", "}", candidate)
                candidate = re.sub(r",\s*]", "]", candidate)
                candidate = re.sub(
                    r'"\s*:\s*None', '": null', candidate
                )
                candidate = re.sub(
                    r'"\s*:\s*True', '": true', candidate
                )
                candidate = re.sub(
                    r'"\s*:\s*False', '": false', candidate
                )
                return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        try:
            code_block = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
            if code_block:
                return json.loads(code_block.group(1).strip())
        except json.JSONDecodeError:
            pass
        return default if default is not None else {}

    def _detect_language(self, text: str) -> str:
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(text.strip())
        if total_chars == 0:
            return "en"
        ratio = chinese_chars / total_chars
        return "zh" if ratio > 0.15 else "en"

    def full_chain(
        self,
        text: str,
        causal_info: Optional[Dict] = None,
        dynamics_info: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        text = text[:3000]
        primary = self.primary_appraisal(text)
        secondary = self.secondary_appraisal(text, primary)
        secondary = self._enforce_primary_secondary_consistency(primary, secondary)
        reappraisal_result = self.reappraisal(primary, secondary, text)
        cp = reappraisal_result.get("corrected_primary", primary.get("primary_appraisal_score", 5))
        cs = reappraisal_result.get("corrected_secondary", secondary.get("secondary_appraisal_score", 5))
        distortions = self.detect_distortions(text, cp, cs)
        try:
            integration = self.integrate(
                primary, secondary, reappraisal_result, distortions, causal_info, dynamics_info
            )
        except Exception as e:
            p_val = reappraisal_result.get("corrected_primary", primary.get("primary_appraisal_score", 5))
            s_val = reappraisal_result.get("corrected_secondary", secondary.get("secondary_appraisal_score", 5))
            social = secondary.get("social_support_perceived", 5)
            n_dist = len(distortions)
            integration = {
                "cusp_proxies": {
                    "a_proxy": (p_val - s_val) / 10.0,
                    "b_proxy": (s_val * social) / 100.0 - 0.5,
                    "c_proxy": social * (11 - n_dist) / 100.0,
                },
                "explanation": f"Integration fallback (error: {e})",
                "intervention": "Based on appraisal scores, consider professional consultation.",
            }
        return {
            "primary_appraisal": primary,
            "secondary_appraisal": secondary,
            "reappraisal": reappraisal_result,
            "cognitive_distortions": distortions,
            "cusp_proxies": integration["cusp_proxies"],
            "explanation": integration["explanation"],
            "intervention": integration["intervention"],
        }

    def primary_appraisal(self, text: str) -> Dict:
        lang = self._detect_language(text)
        if lang == "zh":
            prompt = (
                "你是一名临床心理学家，正在执行Lazarus初级评估（Primary Appraisal）。\n"
                "你的任务是分析以下文本中的威胁识别。\n\n"
                "请分析文本并识别：\n"
                "1. 是否识别到威胁（true/false）\n"
                "2. 威胁类型（如：身体威胁、心理威胁、社会威胁、存在性威胁）\n"
                "3. 威胁强度，1-10分（1=极低，10=极高）\n"
                "4. 简短叙述描述感知到的威胁\n"
                "5. 初级评估得分，1-10分（1=无威胁，10=严重威胁）\n\n"
                "仅以以下JSON格式回复：\n"
                "{\n"
                '  "threat_identified": true,\n'
                '  "threat_type": "string",\n'
                '  "threat_intensity": number,\n'
                '  "threat_narrative": "string",\n'
                '  "primary_appraisal_score": number\n'
                "}\n\n"
                "待分析文本：\n"
                f"{text}\n\n"
                "JSON回复："
            )
        else:
            prompt = (
                "You are a clinical psychologist performing a Lazarus Primary Appraisal for mental health screening. "
                "Your task is to evaluate the following text for clinical-level threat to mental wellbeing.\n\n"
                "IMPORTANT DISTINCTIONS:\n"
                "- Score 1-3: No clinical threat. Normal life stress, everyday complaints, casual mentions of mood, "
                "or general life updates that do NOT indicate psychological distress.\n"
                "- Score 4-5: Mild stress. Some concern but within normal range; the person is coping adequately.\n"
                "- Score 6-7: Moderate clinical concern. Clear signs of psychological distress such as persistent "
                "sadness, anxiety, or difficulty functioning.\n"
                "- Score 8-10: Severe clinical threat. Indicators of major depression, suicidal ideation, "
                "hopelessness, or severe psychological crisis.\n\n"
                "Be CONSERVATIVE: only assign high scores (6+) when there is clear evidence of clinical-level "
                "distress. Do NOT over-interpret casual negativity, humor, or minor frustrations.\n\n"
                "Analyze the text and identify:\n"
                "1. Whether a clinical threat is identified (true/false)\n"
                "2. The type of threat (e.g., physical, psychological, social, existential, none)\n"
                "3. Threat intensity on a scale of 1-10 (1=minimal, 10=extreme)\n"
                "4. A brief narrative describing the perceived threat\n"
                "5. Primary appraisal score on a scale of 1-10 (1=no clinical threat, 10=severe clinical threat)\n\n"
                "Respond ONLY with a JSON object in this exact format:\n"
                "{\n"
                '  "threat_identified": true,\n'
                '  "threat_type": "string",\n'
                '  "threat_intensity": number,\n'
                '  "threat_narrative": "string",\n'
                '  "primary_appraisal_score": number\n'
                "}\n\n"
                "Text to analyze:\n"
                f"{text}\n\n"
                "JSON response:"
            )
        raw = self._generate(prompt)
        parsed = self._parse_json(raw, PRIMARY_DEFAULT.copy())
        if not isinstance(parsed.get("threat_identified"), bool):
            parsed["threat_identified"] = False
        if not isinstance(parsed.get("threat_intensity"), (int, float)):
            parsed["threat_intensity"] = 5
        else:
            parsed["threat_intensity"] = max(1, min(10, int(parsed["threat_intensity"])))
        if not isinstance(parsed.get("primary_appraisal_score"), (int, float)):
            parsed["primary_appraisal_score"] = 5
        else:
            parsed["primary_appraisal_score"] = max(
                1, min(10, int(parsed["primary_appraisal_score"]))
            )
        for key in PRIMARY_DEFAULT:
            if key not in parsed:
                parsed[key] = PRIMARY_DEFAULT[key]
        return parsed

    def secondary_appraisal(self, text: str, primary: Dict) -> Dict:
        lang = self._detect_language(text)
        if lang == "zh":
            prompt = (
                "你是一名临床心理学家，正在执行Lazarus次级评估（Secondary Appraisal）。\n"
                "你的任务是基于以下文本和初级评估结果，评估应对资源和社会支持。\n\n"
                "初级评估结果：\n"
                f"- 是否识别到威胁：{primary.get('threat_identified', False)}\n"
                f"- 威胁类型：{primary.get('threat_type', '未知')}\n"
                f"- 威胁强度：{primary.get('threat_intensity', 5)}/10\n"
                f"- 初级评估得分：{primary.get('primary_appraisal_score', 5)}/10\n\n"
                "请分析文本并评估：\n"
                "1. 可用的应对资源（具体资源列表）\n"
                "2. 应对效能，1-10分（1=极低，10=极高）\n"
                "3. 感知到的社会支持，1-10分（1=极低，10=极高）\n"
                "4. 简短叙述描述可用资源\n"
                "5. 次级评估得分，1-10分（1=无资源，10=资源丰富）\n\n"
                "仅以以下JSON格式回复：\n"
                "{\n"
                '  "coping_resources": ["string"],\n'
                '  "coping_efficacy": number,\n'
                '  "social_support_perceived": number,\n'
                '  "resource_narrative": "string",\n'
                '  "secondary_appraisal_score": number\n'
                "}\n\n"
                "待分析文本：\n"
                f"{text}\n\n"
                "JSON回复："
            )
        else:
            prompt = (
                "You are a clinical psychologist performing a Lazarus Secondary Appraisal for mental health screening. "
                "Your task is to evaluate coping resources and social support based on the following text "
                "and the primary appraisal results.\n\n"
                "IMPORTANT: Score based on EVIDENCE in the text, not assumptions.\n"
                "- Score 8-10: Strong coping. Person explicitly describes effective strategies, "
                "social connections, professional help, or resilience.\n"
                "- Score 5-7: Moderate coping. Some resources mentioned but not comprehensive.\n"
                "- Score 1-4: Low coping. Person describes helplessness, isolation, lack of support, "
                "or inability to manage stress.\n\n"
                "If the text does NOT indicate psychological distress (low primary threat), "
                "assume the person has adequate coping unless evidence suggests otherwise.\n\n"
                "Primary appraisal results:\n"
                f"- Threat identified: {primary.get('threat_identified', False)}\n"
                f"- Threat type: {primary.get('threat_type', 'unknown')}\n"
                f"- Threat intensity: {primary.get('threat_intensity', 5)}/10\n"
                f"- Primary appraisal score: {primary.get('primary_appraisal_score', 5)}/10\n\n"
                "Analyze the text and assess:\n"
                "1. Available coping resources (list of specific resources)\n"
                "2. Coping efficacy on a scale of 1-10 (1=very low, 10=very high)\n"
                "3. Perceived social support on a scale of 1-10 (1=very low, 10=very high)\n"
                "4. A brief narrative describing available resources\n"
                "5. Secondary appraisal score on a scale of 1-10 (1=no resources, 10=abundant resources)\n\n"
                "Respond ONLY with a JSON object in this exact format:\n"
                "{\n"
                '  "coping_resources": ["string"],\n'
                '  "coping_efficacy": number,\n'
                '  "social_support_perceived": number,\n'
                '  "resource_narrative": "string",\n'
                '  "secondary_appraisal_score": number\n'
                "}\n\n"
                "Text to analyze:\n"
                f"{text}\n\n"
                "JSON response:"
            )
        raw = self._generate(prompt)
        parsed = self._parse_json(raw, SECONDARY_DEFAULT.copy())
        if not isinstance(parsed.get("coping_resources"), list):
            parsed["coping_resources"] = []
        if not isinstance(parsed.get("coping_efficacy"), (int, float)):
            parsed["coping_efficacy"] = 5
        else:
            parsed["coping_efficacy"] = max(
                1, min(10, int(parsed["coping_efficacy"]))
            )
        if not isinstance(parsed.get("social_support_perceived"), (int, float)):
            parsed["social_support_perceived"] = 5
        else:
            parsed["social_support_perceived"] = max(
                1, min(10, int(parsed["social_support_perceived"]))
            )
        if not isinstance(parsed.get("secondary_appraisal_score"), (int, float)):
            parsed["secondary_appraisal_score"] = 5
        else:
            parsed["secondary_appraisal_score"] = max(
                1, min(10, int(parsed["secondary_appraisal_score"]))
            )
        for key in SECONDARY_DEFAULT:
            if key not in parsed:
                parsed[key] = SECONDARY_DEFAULT[key]
        return parsed

    def _enforce_primary_secondary_consistency(
        self, primary: Dict, secondary: Dict
    ) -> Dict:
        p_score = primary.get("primary_appraisal_score", 5)
        s_score = secondary.get("secondary_appraisal_score", 5)
        if p_score <= 3 and s_score < 6:
            secondary["secondary_appraisal_score"] = max(s_score, 7)
            secondary["coping_efficacy"] = max(
                secondary.get("coping_efficacy", 5), 6
            )
            secondary["social_support_perceived"] = max(
                secondary.get("social_support_perceived", 5), 6
            )
        elif p_score >= 7 and s_score > 5:
            secondary["secondary_appraisal_score"] = min(s_score, 4)
            secondary["coping_efficacy"] = min(
                secondary.get("coping_efficacy", 5), 5
            )
            secondary["social_support_perceived"] = min(
                secondary.get("social_support_perceived", 5), 5
            )
        return secondary

    def reappraisal(self, primary: Dict, secondary: Dict, text: str) -> Dict:
        p = primary.get("primary_appraisal_score", 5)
        s = secondary.get("secondary_appraisal_score", 5)
        lang = self._detect_language(text)

        if lang == "zh":
            prompt = (
                "你是一名临床心理学家，正在执行Lazarus重新评估（Reappraisal）。\n"
                "你的任务是审视之前的初级和次级评估结果，判断是否存在认知偏差导致的过度评估或低估，"
                "并给出校正后的分数。\n\n"
                "初级评估结果：\n"
                f"- 是否识别到威胁：{primary.get('threat_identified', False)}\n"
                f"- 威胁类型：{primary.get('threat_type', '未知')}\n"
                f"- 威胁强度：{primary.get('threat_intensity', 5)}/10\n"
                f"- 初级评估得分：{p}/10\n\n"
                "次级评估结果：\n"
                f"- 应对效能：{secondary.get('coping_efficacy', 5)}/10\n"
                f"- 感知社会支持：{secondary.get('social_support_perceived', 5)}/10\n"
                f"- 次级评估得分：{s}/10\n\n"
                "请重新审视评估结果，考虑以下可能性：\n"
                "1. 威胁是否被高估（如灾难化思维导致威胁评分偏高）？\n"
                "2. 应对资源是否被低估（如忽视已有支持系统）？\n"
                "3. 威胁是否被低估（如否认机制导致忽视真实风险）？\n"
                "4. 应对资源是否被高估（如不切实际的乐观）？\n\n"
                "仅以以下JSON格式回复：\n"
                "{\n"
                '  "rollback_needed": true/false,\n'
                '  "rollback_reason": "string",\n'
                '  "corrected_primary": number,\n'
                '  "corrected_secondary": number,\n'
                '  "lazarus_consistency": true/false,\n'
                '  "expected_stress": number\n'
                "}\n\n"
                "待重新评估的文本：\n"
                f"{text}\n\n"
                "JSON回复："
            )
        else:
            prompt = (
                "You are a clinical psychologist performing a Lazarus Reappraisal. "
                "Your task is to critically review the primary and secondary appraisal results, "
                "identify potential cognitive biases that may have led to over- or under-estimation, "
                "and provide SMALL corrected scores.\n\n"
                "IMPORTANT: Make only MINOR adjustments (1-2 points max). "
                "Do NOT drastically change scores unless there is overwhelming evidence. "
                "If the initial appraisal seems reasonable, keep the corrected scores the same.\n\n"
                "Primary appraisal results:\n"
                f"- Threat identified: {primary.get('threat_identified', False)}\n"
                f"- Threat type: {primary.get('threat_type', 'unknown')}\n"
                f"- Threat intensity: {primary.get('threat_intensity', 5)}/10\n"
                f"- Primary appraisal score: {p}/10\n\n"
                "Secondary appraisal results:\n"
                f"- Coping efficacy: {secondary.get('coping_efficacy', 5)}/10\n"
                f"- Perceived social support: {secondary.get('social_support_perceived', 5)}/10\n"
                f"- Secondary appraisal score: {s}/10\n\n"
                "Please re-evaluate considering the following possibilities:\n"
                "1. Is the threat overestimated (e.g., catastrophizing inflating threat scores)?\n"
                "2. Are coping resources underestimated (e.g., overlooking existing support)?\n"
                "3. Is the threat underestimated (e.g., denial mechanisms ignoring real risks)?\n"
                "4. Are coping resources overestimated (e.g., unrealistic optimism)?\n\n"
                "Respond ONLY with a JSON object:\n"
                "{\n"
                '  "rollback_needed": true/false,\n'
                '  "rollback_reason": "string",\n'
                '  "corrected_primary": number,\n'
                '  "corrected_secondary": number,\n'
                '  "lazarus_consistency": true/false,\n'
                '  "expected_stress": number\n'
                "}\n\n"
                "Text to re-evaluate:\n"
                f"{text}\n\n"
                "JSON response:"
            )

        REAPPRAISAL_DEFAULT = {
            "rollback_needed": False,
            "rollback_reason": "",
            "corrected_primary": p,
            "corrected_secondary": s,
            "lazarus_consistency": True,
            "expected_stress": p * s / 10.0,
        }

        raw = self._generate(prompt)
        parsed = self._parse_json(raw, REAPPRAISAL_DEFAULT.copy())

        if not isinstance(parsed.get("rollback_needed"), bool):
            parsed["rollback_needed"] = False
        if not isinstance(parsed.get("rollback_reason"), str):
            parsed["rollback_reason"] = ""
        corrected_primary = parsed.get("corrected_primary", p)
        if not isinstance(corrected_primary, (int, float)):
            corrected_primary = p
        corrected_primary = max(1, min(10, int(corrected_primary)))
        corrected_secondary = parsed.get("corrected_secondary", s)
        if not isinstance(corrected_secondary, (int, float)):
            corrected_secondary = s
        corrected_secondary = max(1, min(10, int(corrected_secondary)))
        if not isinstance(parsed.get("lazarus_consistency"), bool):
            parsed["lazarus_consistency"] = not parsed["rollback_needed"]

        parsed["corrected_primary"] = corrected_primary
        parsed["corrected_secondary"] = corrected_secondary
        parsed["expected_stress"] = corrected_primary * corrected_secondary / 10.0

        parsed = self._validate_reappraisal_deterministic(
            parsed, primary, secondary
        )

        p_orig = primary.get("primary_appraisal_score", 5)
        s_orig = secondary.get("secondary_appraisal_score", 5)
        p_corr_final = parsed["corrected_primary"]
        s_corr_final = parsed["corrected_secondary"]
        if abs(p_corr_final - p_orig) > 2:
            parsed["corrected_primary"] = max(
                p_orig - 2, min(p_orig + 2, p_corr_final)
            )
        if abs(s_corr_final - s_orig) > 2:
            parsed["corrected_secondary"] = max(
                s_orig - 2, min(s_orig + 2, s_corr_final)
            )
        parsed["expected_stress"] = (
            parsed["corrected_primary"] * parsed["corrected_secondary"] / 10.0
        )

        return parsed

    def _validate_reappraisal_deterministic(
        self, reappraisal: Dict, primary: Dict, secondary: Dict
    ) -> Dict:
        p_orig = primary.get("primary_appraisal_score", 5)
        s_orig = secondary.get("secondary_appraisal_score", 5)
        p_corr = reappraisal.get("corrected_primary", p_orig)
        s_corr = reappraisal.get("corrected_secondary", s_orig)
        threat_high = p_orig >= 7
        threat_low = p_orig <= 3
        coping_high = s_orig >= 7
        coping_low = s_orig <= 3
        coping_mid = not coping_high and not coping_low
        violations = []
        if threat_high and p_corr > p_orig:
            violations.append(
                "Lazarus constraint violated: high threat should not increase "
                f"via reappraisal, but corrected_primary increased "
                f"from {p_orig} to {p_corr}"
            )
        if coping_low and s_corr < s_orig:
            violations.append(
                "Lazarus constraint violated: low coping should not decrease "
                f"further via reappraisal, but corrected_secondary decreased "
                f"from {s_orig} to {s_corr}"
            )
        if threat_high and coping_low and p_corr < p_orig - 2:
            violations.append(
                "Lazarus constraint violated: high threat + low coping (high stress) "
                "should not drastically reduce threat without new evidence, but "
                f"corrected_primary dropped from {p_orig} to {p_corr}"
            )
        if threat_low and coping_high and s_corr > s_orig + 2:
            violations.append(
                "Lazarus constraint violated: low threat + high coping (low stress) "
                "should not drastically inflate coping without new evidence, but "
                f"corrected_secondary increased from {s_orig} to {s_corr}"
            )
        stress_orig = p_orig * s_orig / 10.0
        stress_corr = p_corr * s_corr / 10.0
        if threat_high and coping_low and stress_corr < stress_orig * 0.5:
            violations.append(
                "Lazarus constraint violated: high-stress state should not "
                f"halve expected stress without justification; stress changed "
                f"from {stress_orig:.1f} to {stress_corr:.1f}"
            )
        if violations:
            p_fixed = p_corr
            s_fixed = s_corr
            if threat_high and p_corr > p_orig:
                p_fixed = max(1, p_orig - 1)
            if coping_low and s_corr < s_orig:
                s_fixed = min(10, s_orig + 1)
            if threat_high and coping_low and p_corr < p_orig - 2:
                p_fixed = p_orig
            if threat_low and coping_high and s_corr > s_orig + 2:
                s_fixed = s_orig
            reappraisal["corrected_primary"] = p_fixed
            reappraisal["corrected_secondary"] = s_fixed
            reappraisal["expected_stress"] = p_fixed * s_fixed / 10.0
            reappraisal["rollback_needed"] = True
            reappraisal["rollback_reason"] = "; ".join(violations)
            reappraisal["lazarus_consistency"] = False
            reappraisal["deterministic_correction_applied"] = True
            reappraisal["llm_original_corrected_primary"] = p_corr
            reappraisal["llm_original_corrected_secondary"] = s_corr
        else:
            reappraisal["lazarus_consistency"] = True
            reappraisal["deterministic_correction_applied"] = False
        return reappraisal

    def detect_distortions(
        self, text: str, corrected_primary: float, corrected_secondary: float
    ) -> List[Dict]:
        lang = self._detect_language(text)
        if lang == "zh":
            prompt = (
                "你是一名受过Aaron Beck认知行为疗法（CBT）训练的临床心理学家。\n"
                "你的任务是使用ABC模型检测以下文本中的认知扭曲。\n\n"
                "背景：该个体的校正后初级评估得分为"
                f"{corrected_primary}/10，校正后次级评估得分为{corrected_secondary}/10。\n\n"
                "仅检测以下5种认知扭曲（如存在）：\n"
                "1. catastrophizing（灾难化）——预期最坏结果\n"
                "2. overgeneralization（过度概括）——从单一事件得出广泛结论\n"
                "3. all_or_nothing（全或无思维）——以非黑即白的方式看待事物\n"
                "4. emotional_reasoning（情绪推理）——认为感受反映了现实\n"
                "5. personalization（个人化）——将超出自己控制的事件归咎于自己\n\n"
                "对每个检测到的扭曲，提供：\n"
                "- type: 上述5种类型之一\n"
                "- severity: 1-5分（1=轻度，5=重度）\n"
                "- evidence: 支持该检测的文本中的具体引用或转述\n\n"
                "仅以JSON数组回复：\n"
                "[\n"
                "  {\n"
                '    "type": "catastrophizing",\n'
                '    "severity": number,\n'
                '    "evidence": "string"\n'
                "  }\n"
                "]\n\n"
                "如未检测到扭曲，返回空数组：[]\n\n"
                "待分析文本：\n"
                f"{text}\n\n"
                "JSON回复："
            )
        else:
            prompt = (
                "You are a clinical psychologist trained in Aaron Beck's Cognitive Behavioral Therapy (CBT). "
                "Your task is to detect cognitive distortions in the following text using the ABC model.\n\n"
                "Context: The person's corrected primary appraisal score is "
                f"{corrected_primary}/10 and corrected secondary appraisal score is {corrected_secondary}/10.\n\n"
                "IMPORTANT: Only detect distortions when there is CLEAR EVIDENCE in the text. "
                "If the text does not show obvious distorted thinking patterns, return an empty array []. "
                "Do NOT detect distortions in normal emotional expressions or reasonable concerns.\n\n"
                "Detect ONLY the following 5 types of cognitive distortions if present:\n"
                "1. catastrophizing - expecting the worst possible outcome\n"
                "2. overgeneralization - drawing broad conclusions from a single event\n"
                "3. all_or_nothing - seeing things in black-and-white categories\n"
                "4. emotional_reasoning - believing that feelings reflect reality\n"
                "5. personalization - blaming oneself for events outside one's control\n\n"
                "For EACH detected distortion, provide:\n"
                "- type: one of the 5 types listed above\n"
                "- severity: 1-5 scale (1=mild, 5=severe)\n"
                "- evidence: specific quote or paraphrase from the text supporting this detection\n\n"
                "Respond ONLY with a JSON array:\n"
                "[\n"
                "  {\n"
                '    "type": "catastrophizing",\n'
                '    "severity": number,\n'
                '    "evidence": "string"\n'
                "  }\n"
                "]\n\n"
                "If no distortions are detected, return an empty array: []\n\n"
                "Text to analyze:\n"
                f"{text}\n\n"
                "JSON response:"
            )
        raw = self._generate(prompt)
        try:
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start >= 0 and end > start:
                candidate = raw[start:end]
                candidate = re.sub(r",\s*]", "]", candidate)
                parsed = json.loads(candidate)
            else:
                parsed = self._parse_json(raw, {})
                if isinstance(parsed, dict):
                    for key in ["distortions", "cognitive_distortions", "results"]:
                        if key in parsed and isinstance(parsed[key], list):
                            parsed = parsed[key]
                            break
                    else:
                        parsed = DISTORTION_DEFAULT
                else:
                    parsed = DISTORTION_DEFAULT
        except json.JSONDecodeError:
            parsed = DISTORTION_DEFAULT
        if not isinstance(parsed, list):
            parsed = DISTORTION_DEFAULT
        validated = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            dtype = item.get("type", "")
            if dtype not in DISTORTION_TYPES:
                continue
            severity = item.get("severity", 3)
            if not isinstance(severity, (int, float)):
                severity = 3
            severity = max(1, min(5, int(severity)))
            evidence = item.get("evidence", "")
            if not isinstance(evidence, str):
                evidence = str(evidence)
            validated.append({"type": dtype, "severity": severity, "evidence": evidence})
        return validated

    def integrate(
        self,
        primary: Dict,
        secondary: Dict,
        reappraisal: Dict,
        distortions: List[Dict],
        causal_info: Optional[Dict],
        dynamics_info: Optional[Dict],
    ) -> Dict:
        p = reappraisal["corrected_primary"]
        s = reappraisal["corrected_secondary"]
        social = secondary.get("social_support_perceived", 5)
        n_distortions = len(distortions)
        a_proxy = (p - s) / 10.0
        b_proxy = (s * social) / 100.0 - 0.5
        c_proxy = social * (11 - n_distortions) / 100.0

        prompt = (
            "You are a clinical psychologist integrating a multi-step Lazarus cognitive appraisal "
            "with Cusp catastrophe theory dynamics for mental health assessment.\n\n"
            "Assessment results:\n"
            f"- Primary appraisal score (corrected): {p}/10\n"
            f"- Secondary appraisal score (corrected): {s}/10\n"
            f"- Perceived social support: {social}/10\n"
            f"- Number of cognitive distortions: {n_distortions}\n"
            f"- Distortion types: {', '.join(d.get('type', 'unknown') for d in distortions) if distortions else 'none'}\n"
            f"- Cusp asymmetry proxy (a): {a_proxy:.3f}\n"
            f"- Cusp bifurcation proxy (b): {b_proxy:.3f}\n"
            f"- Cusp self-regulation proxy (c): {c_proxy:.3f}\n"
        )
        if causal_info:
            prompt += (
                f"\nCausal network info:\n"
                f"- Central symptoms: {causal_info.get('central_symptoms', [])}\n"
                f"- Top feedback loops: {len(causal_info.get('top_loops', []))} detected\n"
            )
        if dynamics_info:
            prompt += (
                f"\nDynamics info:\n"
                f"- Resilience reserve: {dynamics_info.get('resilience_reserve', 'N/A')}\n"
                f"- Critical distance: {dynamics_info.get('critical_distance', 'N/A')}\n"
                f"- Tipping point warning: {dynamics_info.get('tipping_point_warning', False)}\n"
            )
        prompt += (
            "\nProvide:\n"
            "1. A comprehensive explanation of the person's mental state based on the Cusp catastrophe model, "
            "Lazarus appraisal theory, and cognitive distortion analysis. Explain what the proxy values mean "
            "in terms of the person's psychological state.\n"
            "2. Personalized intervention recommendations based on the identified distortions, "
            "central symptoms, and dynamics state.\n\n"
            "Respond ONLY with a JSON object:\n"
            "{\n"
            '  "explanation": "string",\n'
            '  "intervention": "string"\n'
            "}\n\n"
            "JSON response:"
        )
        raw = self._generate(prompt)
        parsed = self._parse_json(raw, {"explanation": "", "intervention": ""})
        explanation = parsed.get("explanation", "")
        intervention = parsed.get("intervention", "")
        if not isinstance(explanation, str):
            explanation = str(explanation)
        if not isinstance(intervention, str):
            intervention = str(intervention)

        return {
            "cusp_proxies": {"a_proxy": a_proxy, "b_proxy": b_proxy, "c_proxy": c_proxy},
            "explanation": explanation,
            "intervention": intervention,
        }
