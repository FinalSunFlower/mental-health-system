"""
Layer 3: Lazarus Cognitive Appraisal Chain with LLM.
Implements the Lazarus transactional stress theory with primary appraisal,
secondary appraisal, reappraisal, cognitive distortion detection, and
CUSP parameter proxy extraction. Includes CoVe verification and self-critique.
"""
import json
import re
import copy
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter


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

RISK_KEYWORDS_HIGH = {
    "suicid": 3, "kill myself": 3, "end it all": 3, "want to die": 3,
    "hopeless": 2, "worthless": 2, "better off dead": 2, "can't go on": 2,
    "no reason to live": 2, "ending my life": 2,
    "depressed every day": 2, "can't get out of bed": 2,
    "nothing matters": 1.5, "completely alone": 1.5, "no future": 1.5,
}

RISK_KEYWORDS_MODERATE = {
    "can't sleep": 1.5, "insomnia": 1.5, "no appetite": 1.5,
    "crying every day": 1.5, "always sad": 1.5, "exhausted all the time": 1.5,
    "can't focus": 1, "don't enjoy anything": 1, "feel empty": 1,
    "overwhelmed": 1, "anxious all the time": 1, "panic": 1,
}


class LazarusAppraisalChain:
    def __init__(
        self,
        model_name: str = r"D:\Models\huggingface\Qwen3.5-2B",
        device: str = "cuda",
        load_in_4bit: bool = True,
        max_new_tokens: int = 768,
        temperature: float = 0.25,
        use_cove: bool = True,
        use_self_critique: bool = True,
        use_risk_sensitive: bool = True,
        n_verification_questions: int = 3,
        max_critique_rounds: int = 2,
        risk_floor: float = 0.15,
    ):
        self.model_name = model_name
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.use_cove = use_cove
        self.use_self_critique = use_self_critique
        self.use_risk_sensitive = use_risk_sensitive
        self.n_verification_questions = n_verification_questions
        self.max_critique_rounds = max_critique_rounds
        self.risk_floor = risk_floor
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

    def _generate(self, prompt: str, temperature: float = None) -> str:
        self._ensure_model()
        import torch

        temp = temperature if temperature is not None else self.temperature
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
                    temperature=max(temp, 1e-5) if temp > 0 else 1e-5,
                    do_sample=temp > 0,
                    top_p=0.92,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            generated_ids = outputs[0][input_len:]
            raw = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            return raw
        except UnicodeDecodeError:
            try:
                return self.tokenizer.decode(generated_ids, skip_special_tokens=True).encode("utf-8", errors="replace").decode("utf-8")
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
                for pattern, replacement in [
                    (r",\s*}", "}"), (r",\s*]", "]"),
                    (r'"\s*:\s*None', '": null'),
                    (r'"\s*:\s*True', '": true'),
                    (r'"\s*:\s*False', '": false'),
                ]:
                    candidate = re.sub(pattern, replacement, candidate)
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
        return "zh" if chinese_chars / total_chars > 0.15 else "en"

    def _scan_risk_keywords(self, text: str) -> Tuple[float, List[str]]:
        text_lower = text.lower()
        total_score = 0.0
        found_keywords = []
        for keyword, weight in RISK_KEYWORDS_HIGH.items():
            if keyword in text_lower:
                total_score += weight
                found_keywords.append(f"HIGH:{keyword}({weight})")
        for keyword, weight in RISK_KEYWORDS_MODERATE.items():
            if keyword in text_lower:
                total_score += weight
                found_keywords.append(f"MOD:{keyword}({weight})")
        return min(total_score / 10.0, 1.0), found_keywords

    def primary_appraisal(self, text: str) -> Dict:
        lang = self._detect_language(text)
        
        draft = self._primary_draft_generation(text, lang)
        
        if self.use_cove:
            verified = self._cove_verify(draft, text, lang)
            result = verified
        else:
            result = draft
        
        if self.use_self_critique:
            critiqued = self._self_critique_primary(result, text, lang)
            result = critiqued
        
        if self.use_risk_sensitive:
            result = self._apply_risk_floor(result, text)
        
        return self._validate_primary(result)

    def _primary_draft_generation(self, text: str, lang: str) -> Dict:
        if lang == "zh":
            prompt = (
                "你是资深临床心理学家，执行Lazarus初级评估。\n\n"
                "【评估目标】判断文本反映的心理健康威胁等级(1-10)\n\n"
                "【评分锚点】\n"
                "- 1-3: 无临床威胁。日常抱怨、正常情绪波动\n"
                "- 4-5: 轻度压力。有担忧但功能正常\n"
                "- 6-7: 中度关注。持续困扰、部分功能受损\n"
                "- 8-10: 严重危机。自杀意念、严重功能障碍\n\n"
                "【关键症状指标】\n"
                "自杀/死亡相关表述 → 至少7分\n"
                "持续2周以上的情绪低落+躯体症状 → 至少6分\n"
                "偶发负面情绪但社会功能正常 → 最多4分\n\n"
                "JSON格式:\n"
                '{threat_identified:bool, threat_type:str, threat_intensity:int(1-10), '
                'threat_narrative:str, primary_appraisal_score:int(1-10)}\n\n'
                f"文本:\n{text}\n\nJSON:"
            )
        else:
            prompt = (
                "You are a senior clinical psychologist performing Primary Appraisal.\n\n"
                "TASK: Assess mental health threat level (1-10) from text.\n\n"
                "SCORE ANCHORS:\n"
                "- 1-3: No clinical threat. Daily complaints, normal mood swings.\n"
                "- 4-5: Mild stress. Concerns present but functioning normally.\n"
                "- 6-7: Moderate concern. Persistent distress, some functional impairment.\n"
                "- 8-10: Severe crisis. Suicidal ideation, severe dysfunction.\n\n"
                "CRITICAL INDICATORS (must elevate score):\n"
                "- Any suicidal/death-related language → minimum score 7\n"
                "- Persistent (>2 weeks) low mood + somatic symptoms → minimum score 6\n"
                "- Occasional negativity with intact function → maximum score 4\n\n"
                "Respond with JSON only:\n"
                '{threat_identified:bool, threat_type:str, threat_intensity:int(1-10), '
                'threat_narrative:str, primary_appraisal_score:int(1-10)}\n\n'
                f"Text:\n{text}\n\nJSON:"
            )

        raw = self._generate(prompt)
        parsed = self._parse_json(raw, PRIMARY_DEFAULT.copy())
        parsed["_draft_raw"] = raw[:300]
        return parsed

    def _cove_verify(self, draft: Dict, text: str, lang: str) -> Dict:
        draft_score = draft.get("primary_appraisal_score", 5)
        draft_narrative = draft.get("threat_narrative", "")
        draft_threat = draft.get("threat_identified", False)

        if lang == "zh":
            verify_prompt = (
                "【CoVe验证阶段】检查初级评估结果是否与文本证据一致。\n\n"
                f"原始评分: {draft_score}/10\n"
                f"威胁识别: {draft_threat}\n"
                f"叙述: {draft_narrative}\n\n"
                f"原文:\n{text[:1500]}\n\n"
                "请回答以下验证问题(Y/N+理由):\n"
                f"Q1: 文本是否包含自杀/自伤/死亡相关表述？如果有，{draft_score}分是否合理？\n"
                f"Q2: 文本是否描述持续的情绪低落或功能障碍？如果有，{draft_score}分是否合理？\n"
                f"Q3: 如果评分为{draft_score}，是否意味着'无临床威胁'(≤3)或'轻度'(≤5)？这与文本内容矛盾吗？\n\n"
                "最终结论(JSON):\n"
                "{is_consistent:bool, verified_score:int, discrepancy:str, correction_needed:bool}"
            )
        else:
            verify_prompt = (
                "[CoVe Verification] Check if assessment is consistent with textual evidence.\n\n"
                f"Draft score: {draft_score}/10\n"
                f"Threat identified: {draft_threat}\n"
                f"Narrative: {draft_narrative}\n\n"
                f"Original text:\n{text[:1500]}\n\n"
                "Answer these verification questions (Y/N + reasoning):\n"
                f"Q1: Does text contain suicide/self-harm/death language? If yes, is score={draft_score} justified?\n"
                f"Q2: Does text describe persistent low mood or dysfunction? If yes, is score={draft_score} justified?\n"
                f"Q3: Score={draft_score} implies {'NO clinical threat' if draft_score <= 3 else 'mild' if draft_score <= 5 else 'moderate-severe'}. "
                f"Does this contradict the text?\n\n"
                "Final verdict (JSON):\n"
                "{is_consistent:bool, verified_score:int, discrepancy:str, correction_needed:bool}"
            )

        raw_verify = self._generate(verify_prompt)
        verify_result = self._parse_json(raw_verify, {"is_consistent": True, "verified_score": draft_score, "discrepancy": "", "correction_needed": False})

        corrected = copy.deepcopy(draft)
        corrected["_cove_verified"] = True
        corrected["_cove_consistent"] = verify_result.get("is_consistent", True)
        corrected["_cove_discrepancy"] = verify_result.get("discrepancy", "")

        if verify_result.get("correction_needed", False):
            new_score = verify_result.get("verified_score", draft_score)
            if isinstance(new_score, (int, float)):
                corrected["primary_appraisal_score"] = max(1, min(10, int(new_score)))
                corrected["_cve_correction"] = f"{draft_score}->{corrected['primary_appraisal_score']}"

        return corrected

    def _self_critique_primary(self, result: Dict, text: str, lang: str) -> Dict:
        current_score = result.get("primary_appraisal_score", 5)
        risk_score, risk_keywords = self._scan_risk_keywords(text)

        if lang == "zh":
            critique_prompt = (
                "【Self-Critique 自我批判】审查自身评估的合理性。\n\n"
                f"当前评分: {current_score}/10\n"
                f"文本风险关键词扫描: 发现 {len(risk_keywords)} 个风险信号\n"
                f"风险关键词: {risk_keywords[:5] if risk_keywords else '无'}\n"
                f"综合风险值: {risk_score:.2f}/1.0\n\n"
                "批判规则:\n"
                "R1: 若文本含高风险词(自杀/绝望)但评分≤5 → 必须提升评分\n"
                "R2: 若文本含中等风险词(失眠/持续悲伤)且评分≤3 → 应提升至至少5\n"
                "R3: 若评分≥8但文本无明显风险词 → 确认是否有充分依据\n"
                "R4: 评分应与风险关键词密度大致正相关\n\n"
                "输出(JSON):\n"
                "{critique_passed:bool, issues:list, adjusted_score:int|null, confidence:float}"
            )
        else:
            critique_prompt = (
                "[Self-Critique] Scrutinize own assessment for consistency.\n\n"
                f"Current score: {current_score}/10\n"
                f"Risk keyword scan: {len(risk_keywords)} signals found\n"
                f"Keywords: {risk_keywords[:5] if risk_keywords else 'none'}\n"
                f"Aggregate risk: {risk_score:.2f}/1.0\n\n"
                "Critique rules:\n"
                "R1: High-risk words (suicide/hopeless) present but score ≤5 → MUST increase\n"
                "R2: Moderate-risk words (insomnia/persistent sadness) present but score ≤3 → should be ≥5\n"
                "R3: Score ≥8 but no clear risk words → verify strong justification exists\n"
                "R4: Score should correlate with risk keyword density\n\n"
                "Output (JSON):\n"
                "{critique_passed:bool, issues:list, adjusted_score:int|null, confidence:float}"
            )

        raw_critique = self._generate(critique_prompt)
        critique_result = self._parse_json(raw_critique, {"critique_passed": True, "issues": [], "adjusted_score": None, "confidence": 0.8})

        corrected = copy.deepcopy(result)
        corrected["_critiqued"] = True
        corrected["_critique_issues"] = critique_result.get("issues", [])
        corrected["_critique_confidence"] = critique_result.get("confidence", 0.8)

        adj_score = critique_result.get("adjusted_score")
        if adj_score is not None and isinstance(adj_score, (int, float)):
            old = corrected.get("primary_appraisal_score", 5)
            new = max(1, min(10, int(adj_score)))
            corrected["primary_appraisal_score"] = new
            corrected["_critique_adjustment"] = f"{old}->{new}"

        return corrected

    def _apply_risk_floor(self, result: Dict, text: str) -> Dict:
        risk_score, keywords = self._scan_risk_keywords(text)
        current = result.get("primary_appraisal_score", 5)

        floor_score = max(3, round(risk_score * 9 + 1))

        if current < floor_score and risk_score > 0.2:
            old = current
            result["primary_appraisal_score"] = max(current, floor_score)
            result["_risk_floor_applied"] = True
            result["_risk_floor"] = floor_score
            result["_risk_keywords_found"] = [k.split("(")[0] for k in keywords]
            result["_risk_adjustment"] = f"{old}->{result['primary_appraisal_score']}"
        else:
            result["_risk_floor_applied"] = False

        return result

    def _validate_primary(self, parsed: Dict) -> Dict:
        if not isinstance(parsed.get("threat_identified"), bool):
            parsed["threat_identified"] = False
        if not isinstance(parsed.get("threat_intensity"), (int, float)):
            parsed["threat_intensity"] = 5
        else:
            parsed["threat_intensity"] = max(1, min(10, int(parsed["threat_intensity"])))
        if not isinstance(parsed.get("primary_appraisal_score"), (int, float)):
            parsed["primary_appraisal_score"] = 5
        else:
            parsed["primary_appraisal_score"] = max(1, min(10, int(parsed["primary_appraisal_score"])))
        for key in PRIMARY_DEFAULT:
            if key not in parsed:
                parsed[key] = PRIMARY_DEFAULT[key]
        return parsed

    def secondary_appraisal(self, text: str, primary: Dict) -> Dict:
        lang = self._detect_language(text)
        p_score = primary.get("primary_appraisal_score", 5)
        ideal_s = max(1, min(10, 11 - p_score))

        if lang == "zh":
            prompt = (
                "执行Lazarus次级评估——评估应对资源。\n\n"
                f"初级评估得分: {p_score}/10 ({'高威胁' if p_score >= 6 else '低威胁' if p_score <= 4 else '中等'})\n\n"
                "【逆关系约束】威胁越高→应对能力越低\n"
                f"建议次级评分范围: [{max(1, ideal_s-2)}, {min(10, ideal_s+2)}]\n\n"
                "评分标准:\n"
                "- 1-3: 无应对资源，孤立无助\n"
                "- 4-6: 有一些但不稳定\n"
                "- 7-10: 有充足支持系统\n\n"
                "JSON:\n"
                '{coping_resources:list, coping_efficacy:int(1-10), '
                'social_support_perceived:int(1-10), secondary_appraisal_score:int(1-10)}\n\n'
                f"文本:\n{text}\n\nJSON:"
            )
        else:
            prompt = (
                "Perform Lazarus Secondary Appraisal — assess coping resources.\n\n"
                f"Primary appraisal score: {p_score}/10 ({'HIGH threat' if p_score >= 6 else 'LOW threat' if p_score <= 4 else 'MODERATE'})\n\n"
                "INVERSE CONSTRAINT: Higher threat → Lower coping ability\n"
                f"Recommended secondary range: [{max(1, ideal_s-2)}, {min(10, ideal_s+2)}]\n\n"
                "Scoring:\n"
                "- 1-3: No resources, isolated, helpless\n"
                "- 4-6: Some but unstable\n"
                "- 7-10: Strong support system\n\n"
                "JSON:\n"
                '{coping_resources:list, coping_efficacy:int(1-10), '
                'social_support_perceived:int(1-10), secondary_appraisal_score:int(1-10)}\n\n'
                f"Text:\n{text}\n\nJSON:"
            )

        raw = self._generate(prompt)
        parsed = self._parse_json(raw, SECONDARY_DEFAULT.copy())
        parsed = self._validate_secondary(parsed)
        parsed = self._apply_inverse_constraint(primary, parsed)
        return parsed

    def _validate_secondary(self, parsed: Dict) -> Dict:
        if not isinstance(parsed.get("coping_resources"), list):
            parsed["coping_resources"] = []
        for key in ["coping_efficacy", "social_support_perceived", "secondary_appraisal_score"]:
            val = parsed.get(key, 5)
            if not isinstance(val, (int, float)):
                val = 5
            parsed[key] = max(1, min(10, int(val)))
        for key in SECONDARY_DEFAULT:
            if key not in parsed:
                parsed[key] = SECONDARY_DEFAULT[key]
        return parsed

    def _apply_inverse_constraint(self, primary: Dict, secondary: Dict) -> Dict:
        p_score = primary.get("primary_appraisal_score", 5)
        s_score = secondary.get("secondary_appraisal_score", 5)
        ideal_s = max(1, min(10, 11 - p_score))
        deviation = s_score - ideal_s

        if abs(deviation) > 2:
            pull_strength = min(abs(deviation) * 0.18, 0.45)
            corrected_s = s_score - deviation * pull_strength
            secondary["secondary_appraisal_score"] = max(1, min(10, round(corrected_s)))
            secondary["_inverse_corrected"] = True
            secondary["_inverse_amount"] = round(deviation * pull_strength, 2)

        return secondary

    def _enforce_primary_secondary_consistency(self, primary: Dict, secondary: Dict) -> Dict:
        p_score = primary.get("primary_appraisal_score", 5)
        s_score = secondary.get("secondary_appraisal_score", 5)
        ideal = max(1, min(10, 11 - p_score))
        deviation = s_score - ideal

        if abs(deviation) > 3:
            blend = round(0.55 * ideal + 0.45 * s_score)
            secondary["secondary_appraisal_score"] = max(1, min(10, blend))
            if deviation > 0:
                secondary["coping_efficacy"] = min(secondary.get("coping_efficacy", 5), max(1, blend - 1))
            else:
                secondary["coping_efficacy"] = max(secondary.get("coping_efficacy", 5), min(10, blend + 1))
        return secondary

    def reappraisal(self, primary: Dict, secondary: Dict, text: str) -> Dict:
        p = primary.get("primary_appraisal_score", 5)
        s = secondary.get("secondary_appraisal_score", 5)
        lang = self._detect_language(text)

        if lang == "zh":
            prompt = (
                f"Lazarus重新评估。初级={p}/10, 次级={s}/10\n"
                "一致性检查: 高威胁(p≥7)+高应对(s≥6)=异常 | 低威胁(p≤3)+低应对(s≤4)=异常\n"
                "仅微调±1-2分，除非压倒性证据否则保持。\n"
                "{rollback_needed:bool, rollback_reason:str, corrected_primary:int, corrected_secondary:int, "
                "lazarus_consistency:bool, expected_stress:number}\n\n"
                f"文本: {text}\n\nJSON:"
            )
        else:
            prompt = (
                f"Lazarus Reappraisal. Primary={p}/10, Secondary={s}/10\n"
                f"Consistency check: {'CONFLICT' if p >= 7 and s >= 6 else 'OK'} | "
                f"{'CONFLICT' if p <= 3 and s <= 4 else 'OK'}\n"
                "Minor adjustments only (±1-2 pts). Keep if reasonable.\n"
                "{rollback_needed:bool, rollback_reason:str, corrected_primary:int, corrected_secondary:int, "
                "lazarus_consistency:bool, expected_stress:number}\n\n"
                f"Text: {text}\n\nJSON:"
            )

        raw = self._generate(prompt)
        default = {"rollback_needed": False, "rollback_reason": "", "corrected_primary": p,
                     "corrected_secondary": s, "lazarus_consistency": True, "expected_stress": p*s/10}
        parsed = self._parse_json(raw, default)

        cp = max(1, min(10, int(parsed.get("corrected_primary", p))))
        cs = max(1, min(10, int(parsed.get("corrected_secondary", s))))

        if abs(cp - p) > 2:
            cp = max(p - 2, min(p + 2, cp))
        if abs(cs - s) > 2:
            cs = max(s - 2, min(s + 2, cs))

        parsed["corrected_primary"] = cp
        parsed["corrected_secondary"] = cs
        parsed["expected_stress"] = cp * cs / 10.0
        return parsed

    def detect_distortions(self, text: str, cp: float, cs: float) -> List[Dict]:
        lang = self._detect_language(text)
        if lang == "zh":
            prompt = (
                f"检测认知扭曲。初级={cp}/10, 次级={cs}/10\n"
                "类型: catastrophizing/overgeneralization/all_or_nothing/emotional_reasoning/personalization\n"
                '[{"type":"...", "severity":1-5, "evidence":"..."}]\n'
                f"文本: {text}\n\nJSON:"
            )
        else:
            prompt = (
                f"Detect cognitive distortions. P={cp}/10, S={cs}/10\n"
                "Types: catastrophizing/overgeneralization/all_or_nothing/emotional_reasoning/personalization\n"
                '[{"type":"...", "severity":1-5, "evidence":"..."}]\n'
                f"Text: {text}\n\nJSON:"
            )

        raw = self._generate(prompt)
        parsed = self._parse_json(raw, DISTORTION_DEFAULT)
        if isinstance(parsed, list):
            validated = []
            for item in parsed:
                if isinstance(item, dict) and item.get("type") in DISTORTION_TYPES:
                    sev = max(1, min(5, int(item.get("severity", 3))))
                    item["severity"] = sev
                    validated.append(item)
            return validated
        return DISTORTION_DEFAULT

    def integrate(self, primary, secondary, reappraisal, distortions, causal_info, dynamics_info):
        p = reappraisal["corrected_primary"]
        s = reappraisal["corrected_secondary"]
        social = secondary.get("social_support_perceived", 5)
        n_dist = len(distortions)
        a_proxy = round((p - s) / 10.0, 4)
        b_proxy = round((s * social) / 100.0 - 0.5, 4)
        c_proxy = round(social * (11 - n_dist) / 100.0, 4)

        context = f"P={p}/10 S={s}/10 Social={social}/10 Distortions={n_dist} Cusp=[{a_proxy},{b_proxy},{c_proxy}]"
        if causal_info:
            context += f" Loops={len(causal_info.get('top_loops', []))}"
        if dynamics_info:
            context += f" Tipping={'Y' if dynamics_info.get('tipping_point_warning') else 'N'}"

        prompt = (
            "Integrate Lazarus+Cusp assessment into clinical narrative.\n\n"
            f"{context}\n\n"
            "{explanation:str, intervention:str}\n\nJSON:"
        )
        raw = self._generate(prompt)
        parsed = self._parse_json(raw, {"explanation": "", "intervention": ""})
        return {
            "cusp_proxies": {"a_proxy": a_proxy, "b_proxy": b_proxy, "c_proxy": c_proxy},
            "explanation": str(parsed.get("explanation", "")),
            "intervention": str(parsed.get("intervention", "")),
        }

    def full_chain(self, text: str, causal_info=None, dynamics_info=None) -> Dict[str, Any]:
        text = text[:3000]

        primary = self.primary_appraisal(text)
        secondary = self.secondary_appraisal(text, primary)
        raw_sec = secondary.get("secondary_appraisal_score", 5)
        secondary = self._enforce_primary_secondary_consistency(primary, secondary)
        reappraisal_result = self.reappraisal(primary, secondary, text)

        cp = reappraisal_result.get("corrected_primary", primary["primary_appraisal_score"])
        cs = reappraisal_result.get("corrected_secondary", secondary["secondary_appraisal_score"])
        distortions = self.detect_distortions(text, cp, cs)

        ideal_s = max(1, min(10, 11 - cp))
        sec_gap = abs(cs - ideal_s)
        trust = max(0.2, 1.0 - sec_gap / 10.0)

        try:
            integration = self.integrate(primary, secondary, reappraisal_result, distortions, causal_info, dynamics_info)
        except Exception as e:
            integration = {
                "cusp_proxies": {"a_proxy": (cp-cs)/10, "b_proxy": cs*secondary.get("social_support_perceived",5)/100-0.5,
                                  "c_proxy": secondary.get("social_support_perceived",5)*(11-len(distortions))/100},
                "explanation": f"Fallback (error: {e})", "intervention": "Consider professional consultation.",
            }

        proxies = integration["cusp_proxies"]
        proxies["a_proxy"] = round(trust * proxies["a_proxy"] + (1-trust) * ((cp - ideal_s)/10), 4)

        meta_keys = ["_draft_raw", "_cove_verified", "_cove_consistent", "_cove_discrepancy",
                      "_cve_correction", "_critiqued", "_critique_issues", "_critique_confidence",
                      "_critique_adjustment", "_risk_floor_applied", "_risk_floor", "_risk_keywords_found",
                      "_risk_adjustment", "_inverse_corrected", "_inverse_amount"]

        calibration_meta = {
            "trust_factor": round(trust, 4),
            "raw_secondary": raw_sec,
            "enforced_secondary": secondary.get("secondary_appraisal_score", 5),
            "ideal_secondary": ideal_s,
            "secondary_gap": round(sec_gap, 2),
            "corrected_primary": cp,
            "corrected_secondary": cs,
        }
        for k in meta_keys:
            if k in primary:
                calibration_meta[k.lstrip("_")] = primary[k]

        return {
            "primary_appraisal": primary,
            "secondary_appraisal": secondary,
            "reappraisal": reappraisal_result,
            "cognitive_distortions": distortions,
            "cusp_proxies": proxies,
            "explanation": integration["explanation"],
            "intervention": integration["intervention"],
            "calibration_meta": calibration_meta,
        }
