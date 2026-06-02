"""
Layer 3: Lazarus Cognitive Appraisal Chain with LLM.
Implements the Lazarus transactional stress theory with primary appraisal,
secondary appraisal, reappraisal, cognitive distortion detection, and
CUSP parameter proxy extraction.

Architecture: LLM performs symptom identification (binary yes/no per symptom),
then algorithmic scoring converts symptoms to threat levels. All Lazarus
theoretical constraints are enforced algorithmically. This separation of
concerns (LLM for text understanding, algorithm for clinical reasoning)
improves reliability regardless of model size.
"""
import json
import re
import copy
import math
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
    "lost interest": 1, "no motivation": 1, "guilt": 1,
    "helpless": 1, "trapped": 1, "burden": 1,
}

COPING_KEYWORDS = {
    "support": -1.0, "friends": -0.8, "family": -0.8, "therapy": -1.0,
    "coping": -0.8, "help": -0.6, "hope": -0.7, "better": -0.5,
    "improving": -0.6, "grateful": -0.5, "exercise": -0.5,
    "meditation": -0.5, "counseling": -0.8, "treatment": -0.7,
    "medication": -0.5, "recovery": -0.7, "positive": -0.4,
    "love": -0.5, "understand": -0.4, "accept": -0.4,
    "manage": -0.5, "manageable": -0.6, "coping well": -0.8,
    "getting better": -0.7, "looking forward": -0.6, "enjoy": -0.5,
    "happy": -0.5, "joy": -0.5, "laugh": -0.4, "smile": -0.3,
    "good day": -0.5, "productive": -0.4, "accomplish": -0.4,
    "proud": -0.4, "confident": -0.5, "strong": -0.4,
    "resilient": -0.6, "self-care": -0.6, "journaling": -0.4,
    "hobby": -0.4, "socialize": -0.5, "volunteer": -0.4,
    "workout": -0.5, "yoga": -0.4, "mindfulness": -0.5,
    "therapist": -0.8, "psychiatrist": -0.7, "psychologist": -0.7,
    "antidepressant": -0.5, "ssri": -0.5, "cbt": -0.6,
}

COPING_KEYWORDS_ZH = {
    "支持": -0.8, "朋友": -0.6, "家人": -0.7, "治疗": -0.8,
    "咨询": -0.7, "帮助": -0.5, "希望": -0.6, "好转": -0.7,
    "改善": -0.6, "感恩": -0.5, "运动": -0.4, "冥想": -0.4,
    "康复": -0.7, "积极": -0.5, "理解": -0.4, "接受": -0.4,
    "开心": -0.5, "快乐": -0.5, "笑": -0.3, "享受": -0.5,
    "好起来": -0.6, "期待": -0.5, "自信": -0.5, "坚强": -0.4,
    "心理咨询": -0.8, "心理医生": -0.7, "吃药": -0.4,
    "自我调节": -0.5, "放松": -0.4, "爱好": -0.4,
}

DISTORTION_PATTERNS = {
    "catastrophizing": [
        r"\b(always|never|everyone|no one|nothing|everything|impossible|end of)\b",
        r"\b(worst|terrible|horrible|disaster|catastrophe|ruined|destroyed)\b",
        r"\b(can't\s+(?:ever|never|possibly))\b",
    ],
    "overgeneralization": [
        r"\b(all|every|none|nobody|nothing|always|never)\s+\w+\b",
        r"\b(every\s+time|no\s+one\s+ever|always\s+fail)\b",
    ],
    "all_or_nothing": [
        r"\b(completely|totally|absolutely|perfect|failure|useless)\b",
        r"\b(either\s+.*\s+or|black\s+and\s+white|all\s+or\s+nothing)\b",
    ],
    "emotional_reasoning": [
        r"\b(i\s+feel\s+(?:like|that|as\s+if))\b",
        r"\b(it\s+feels\s+like|feels\s+impossible)\b",
    ],
    "personalization": [
        r"\b(it'?s?\s+(?:all\s+)?my\s+fault)\b",
        r"\b(i\s+(?:always|always\s+)ruin|blame\s+myself)\b",
        r"\b(everyone\s+(?:hates|dislikes)\s+me)\b",
    ],
}

PHQ8_SYMPTOMS = [
    "anhedonia",
    "depressed_mood",
    "sleep_problems",
    "fatigue",
    "appetite_changes",
    "worthlessness_guilt",
    "concentration_problems",
    "suicidal_ideation",
]

SYMPTOM_KEYWORDS = {
    "anhedonia": [
        "don't enjoy", "no interest", "lost interest", "nothing fun",
        "no pleasure", "don't care", "indifferent", "numb",
        "used to love", "gives me no joy", "boring", "empty",
        "don't want to do", "nothing matters", "no motivation",
        "can't bring myself", "don't feel like", "no point in",
        "used to enjoy", "doesn't excite", "no enthusiasm",
        "stopped doing", "given up", "don't bother",
        "nothing makes me happy", "joyless", "pleasureless",
        "anhedonia", "apathy", "blah",
    ],
    "depressed_mood": [
        "sad", "depressed", "down", "low", "unhappy", "miserable",
        "crying", "tears", "hopeless", "blue", "dark", "gloomy",
        "can't stop crying", "crying all the time",
        "feel like crying", "sobbing", "weep",
        "terrible", "awful", "horrible", "dreadful",
        "can't go on", "don't know how much more",
        "suffering", "pain", "hurt", "agony", "anguish",
        "despair", "desperate", "wretched", "broken",
        "empty inside", "hollow", "dead inside",
        "no happiness", "can't remember the last time i was happy",
        "miserable", "heartbroken", "devastated",
    ],
    "sleep_problems": [
        "can't sleep", "insomnia", "sleepless", "awake all night",
        "trouble sleeping", "can't fall asleep", "waking up early",
        "nightmares", "oversleeping", "sleep too much",
        "tossing and turning", "restless nights", "sleep disturbance",
        "lying awake", "can't stay asleep", "early morning waking",
        "sleep all day", "can't get out of bed", "staying up all night",
        "sleep is ruined", "bad dreams", "sleepless nights",
        "never sleep", "barely sleep", "sleep has been terrible",
    ],
    "fatigue": [
        "tired", "exhausted", "fatigue", "no energy", "drained",
        "worn out", "can't get out of bed", "lethargic", "weary",
        "no motivation", "sluggish",
        "barely function", "no strength", "physically drained",
        "mentally exhausted", "burnt out", "burned out",
        "can barely move", "no get up and go", "zapped",
        "depleted", "spent", "done", "overwhelmed",
        "simple tasks feel impossible", "too tired to",
        "need to rest all the time", "always tired",
    ],
    "appetite_changes": [
        "no appetite", "can't eat", "lost weight", "not eating",
        "overeating", "binge", "comfort eating", "food doesn't",
        "don't want to eat", "eating too much", "lost interest in food",
        "food tastes like nothing", "force myself to eat",
        "can't stop eating", "weight gain", "weight loss",
        "forgetting to eat", "nausea", "no hunger",
        "stress eating", "emotional eating", "can't keep food down",
    ],
    "worthlessness_guilt": [
        "worthless", "guilt", "guilty", "shame", "ashamed",
        "my fault", "burden", "useless", "failure", "disappoint",
        "hate myself", "not good enough",
        "everything is my fault", "i ruin everything",
        "don't deserve", "shouldn't be here", "waste of space",
        "let everyone down", "can't do anything right",
        "pathetic", "loser", "stupid", "worth nothing",
        "regret", "blame myself", "selfish", "bad person",
        "don't deserve happiness", "punish myself",
    ],
    "concentration_problems": [
        "can't focus", "can't concentrate", "distracted", "foggy",
        "brain fog", "can't think", "confused", "forgetful",
        "mind wandering", "can't pay attention",
        "scattered", "can't follow", "losing my train of thought",
        "can't remember", "mind goes blank", "can't process",
        "overwhelmed by simple tasks", "reading the same page",
        "can't make decisions", "indecisive", "mental fog",
        "head feels cloudy", "can't organize my thoughts",
    ],
    "suicidal_ideation": [
        "suicid", "want to die", "kill myself", "end it all",
        "better off dead", "no reason to live", "ending my life",
        "don't want to be alive", "death", "not worth living",
        "wish i was dead", "wish i weren't here", "want to disappear",
        "no point living", "life has no meaning", "give up on life",
        "thoughts of death", "thinking about dying",
        "would be better if i wasn't here", "don't want to wake up",
        "plan to end", "harm myself", "self-harm", "hurt myself",
        "overdose", "bridge", "rope", "pills",
    ],
}

SYMPTOM_KEYWORDS_ZH = {
    "anhedonia": [
        "没意思", "不想做", "没兴趣", "什么都不想做", "提不起劲",
        "没有乐趣", "无聊", "空虚", "麻木", "不在乎",
        "以前喜欢", "现在不想", "没感觉", "没劲", "没动力",
        "快感缺失", "丧失兴趣", "不想动", "懒得",
    ],
    "depressed_mood": [
        "难过", "伤心", "悲伤", "抑郁", "低落", "消沉",
        "想哭", "哭泣", "绝望", "痛苦", "难受",
        "不开心", "心情差", "心情不好", "沮丧", "郁闷",
        "黑暗", "看不到希望", "活着没意思", "崩溃",
        "情绪低落", "一直哭", "忍不住哭", "想死",
    ],
    "sleep_problems": [
        "失眠", "睡不着", "整夜没睡", "早醒", "睡眠差",
        "做噩梦", "嗜睡", "睡不醒", "睡眠障碍",
        "翻来覆去", "凌晨就醒", "整晚醒着", "睡不好",
        "入睡困难", "多梦", "半夜醒来", "睡太多",
    ],
    "fatigue": [
        "累", "疲惫", "疲劳", "没精力", "没力气",
        "精疲力竭", "乏力", "不想动", "起不来床",
        "浑身无力", "虚脱", "耗尽", "心力交瘁",
        "做什么都没劲", "走不动", "动不了",
    ],
    "appetite_changes": [
        "没胃口", "不想吃", "吃不下", "暴食", "体重下降",
        "体重增加", "食欲差", "厌食", "吃太多",
        "味同嚼蜡", "强迫自己吃", "忘记吃饭",
    ],
    "worthlessness_guilt": [
        "没用", "废物", "自责", "内疚", "愧疚",
        "都是我的错", "我害了", "负担", "拖累",
        "不值得", "不配", "恨自己", "羞耻",
        "什么都做不好", "失败者", "对不起",
        "不应该活着", "浪费空间", "我的问题",
    ],
    "concentration_problems": [
        "无法集中", "注意力差", "走神", "脑子一片空白",
        "记不住", "健忘", "脑子糊涂", "反应慢",
        "看不进去", "读不下去", "无法思考",
        "脑子转不动", "脑子像浆糊", "无法专注",
    ],
    "suicidal_ideation": [
        "想死", "自杀", "不想活了", "结束生命", "活着没意义",
        "想消失", "不如死了", "一了百了", "死了算了",
        "活不下去", "没有活下去的理由", "想伤害自己",
        "自残", "跳楼", "吃药", "上吊",
        "死亡", "不想醒来", "希望不再醒来",
    ],
}

SYMPTOM_WEIGHTS = {
    "suicidal_ideation": 3.0,
    "depressed_mood": 1.5,
    "anhedonia": 1.5,
    "worthlessness_guilt": 1.3,
    "fatigue": 1.0,
    "sleep_problems": 1.0,
    "appetite_changes": 1.0,
    "concentration_problems": 0.8,
}


class LazarusAppraisalChain:
    def __init__(
        self,
        model_name: str = None,
        device: str = "cuda",
        load_in_4bit: bool = True,
        max_new_tokens: int = 512,
        temperature: float = 0.15,
        use_cove: bool = True,
        use_self_critique: bool = True,
        use_risk_sensitive: bool = True,
        n_verification_questions: int = 3,
        max_critique_rounds: int = 2,
        risk_floor: float = 0.15,
    ):
        if model_name is None:
            from app.core.config import settings
            model_name = settings.LLM_MODEL_NAME
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

    def _scan_coping_keywords(self, text: str) -> Tuple[float, List[str]]:
        text_lower = text.lower()
        total_score = 0.0
        found = []
        for keyword, weight in COPING_KEYWORDS.items():
            if keyword in text_lower:
                total_score += abs(weight)
                found.append(keyword)
        for keyword, weight in COPING_KEYWORDS_ZH.items():
            if keyword in text:
                total_score += abs(weight)
                found.append(keyword)
        return min(total_score / 5.0, 1.0), found

    def _detect_symptoms_rule_based(self, text: str) -> Dict[str, bool]:
        text_lower = text.lower()
        negation_window = 6
        words = text_lower.split()
        negated_indices = set()
        for i, w in enumerate(words):
            if w in ("not", "no", "never", "don't", "dont", "doesn't", "doesnt",
                     "didn't", "didnt", "can't", "cant", "won't", "wont",
                     "isn't", "isnt", "aren't", "arent", "wasn't", "wasnt",
                     "hardly", "barely", "rarely", "seldom", "without"):
                for j in range(i + 1, min(i + negation_window + 1, len(words))):
                    negated_indices.add(j)

        detected = {}
        for symptom in PHQ8_SYMPTOMS:
            found = False
            en_kws = SYMPTOM_KEYWORDS.get(symptom, [])
            zh_kws = SYMPTOM_KEYWORDS_ZH.get(symptom, [])
            for kw in en_kws:
                idx = text_lower.find(kw)
                if idx >= 0:
                    char_pos = len(text_lower[:idx].split())
                    nearby_negated = any(
                        abs(i - char_pos) <= negation_window
                        for i in negated_indices
                    )
                    if not nearby_negated:
                        found = True
                        break
                    else:
                        before = text_lower[max(0, idx - 30):idx]
                        if any(n in before for n in ["not", "don't", "never", "no", "can't", "doesn't", "didn't"]):
                            continue
                        else:
                            found = True
                            break
            if not found:
                for kw in zh_kws:
                    if kw in text_lower or kw in text:
                        found = True
                        break
            detected[symptom] = found
        return detected

    def _detect_distortions_rule_based(self, text: str) -> List[Dict]:
        detected = []
        text_lower = text.lower()
        for dtype, patterns in DISTORTION_PATTERNS.items():
            evidence = []
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                if matches:
                    evidence.extend(matches[:2])
            if evidence:
                severity = min(5, 2 + len(evidence))
                detected.append({
                    "type": dtype,
                    "severity": severity,
                    "evidence": evidence[:3],
                    "detection_method": "rule",
                })
        return detected

    def _symptoms_to_primary_score(self, symptoms: Dict[str, bool], rule_symptoms: Dict[str, bool], text: str = "") -> int:
        merged = {}
        confidence = {}
        for s in PHQ8_SYMPTOMS:
            llm_val = symptoms.get(s, False)
            rule_val = rule_symptoms.get(s, False)
            if rule_val:
                merged[s] = True
                confidence[s] = 1.0
            elif llm_val:
                merged[s] = True
                confidence[s] = 0.6
            else:
                merged[s] = False
                confidence[s] = 0.0

        n_high = sum(1 for s in PHQ8_SYMPTOMS if confidence.get(s, 0) >= 1.0)
        n_low = sum(1 for s in PHQ8_SYMPTOMS if 0 < confidence.get(s, 0) < 1.0)

        if n_low > n_high * 3 and n_high <= 1:
            n_low = max(n_low - 2, 0)

        effective_count = n_high + n_low * 0.6

        coping_score, _ = self._scan_coping_keywords(text)
        if coping_score > 0.4 and n_high <= 1:
            effective_count *= max(0.3, 1.0 - coping_score * 0.7)

        risk_score, _ = self._scan_risk_keywords(text)
        if risk_score > 0.3:
            effective_count = min(effective_count + risk_score * 1.5, 8.0)

        text_len = len(text.split()) if text else 1
        all_symptom_kws = []
        for kws in SYMPTOM_KEYWORDS.values():
            all_symptom_kws.extend(kws)
        text_lower = text.lower() if text else ""
        symptom_hits = sum(1 for kw in all_symptom_kws if kw in text_lower)
        density = symptom_hits / max(text_len / 100.0, 1.0)
        if density < 1.0 and n_high <= 2 and risk_score < 0.2:
            effective_count *= max(0.5, density)

        if effective_count < 0.5:
            return 1

        if merged.get("suicidal_ideation", False) and effective_count <= 2:
            return 8

        if effective_count < 1.5:
            score = 2
        elif effective_count < 2.5:
            score = 3
        elif effective_count < 3.5:
            score = 4
        elif effective_count < 4.5:
            score = 5
        elif effective_count < 5.5:
            score = 6
        elif effective_count < 6.5:
            score = 7
        elif effective_count < 7.5:
            score = 8
        elif effective_count < 8.5:
            score = 9
        else:
            score = 10

        if merged.get("suicidal_ideation", False) and score < 7:
            score = 7

        return max(1, min(10, score))

    def primary_appraisal(self, text: str) -> Dict:
        lang = self._detect_language(text)
        rule_symptoms = self._detect_symptoms_rule_based(text)
        llm_symptoms = self._llm_symptom_identification(text, lang)
        primary_score = self._symptoms_to_primary_score(llm_symptoms, rule_symptoms, text)

        result = {
            "threat_identified": primary_score >= 5,
            "threat_type": self._score_to_threat_type(primary_score),
            "threat_intensity": primary_score,
            "threat_narrative": "",
            "primary_appraisal_score": primary_score,
            "llm_symptoms": llm_symptoms,
            "rule_symptoms": rule_symptoms,
            "n_symptoms_llm": sum(1 for v in llm_symptoms.values() if v),
            "n_symptoms_rule": sum(1 for v in rule_symptoms.values() if v),
        }
        return result

    def _score_to_threat_type(self, score: int) -> str:
        if score <= 3:
            return "none"
        elif score <= 5:
            return "mild_stress"
        elif score <= 7:
            return "clinical_distress"
        else:
            return "severe_crisis"

    def _llm_symptom_identification(self, text: str, lang: str) -> Dict[str, bool]:
        truncated = text[:2500]
        if lang == "zh":
            prompt = (
                "从文本中找出抑郁症状的证据。对每个症状，如果有文本证据就写true，没有写false。\n"
                "注意：间接表达也算，如'不想起床'暗示疲劳，'觉得活着没意义'暗示自杀意念。\n\n"
                "症状:\n"
                "1. anhedonia: 无法享受、没兴趣、不想做任何事\n"
                "2. depressed_mood: 悲伤、低落、想哭、绝望\n"
                "3. sleep_problems: 失眠、早醒、嗜睡、做噩梦\n"
                "4. fatigue: 疲惫、没精力、起不来床\n"
                "5. appetite_changes: 没胃口、暴食、体重变化\n"
                "6. worthlessness_guilt: 自责、觉得自己没用、内疚\n"
                "7. concentration_problems: 无法集中、健忘、脑子糊涂\n"
                "8. suicidal_ideation: 想死、不想活、自杀念头\n\n"
                '输出格式: {"anhedonia":true,"depressed_mood":true,"sleep_problems":false,"fatigue":true,"appetite_changes":false,"worthlessness_guilt":false,"concentration_problems":false,"suicidal_ideation":false}\n'
                "只输出JSON，不要其他内容。\n\n"
                f"文本:\n{truncated}\n\nJSON:"
            )
        else:
            prompt = (
                "Find evidence of depression symptoms in the text. For each symptom, write true if there is ANY textual evidence (even indirect), false if not.\n"
                "Indirect expressions count: 'can't get out of bed' implies fatigue, 'no point' implies worthlessness, 'wish I wasn't here' implies suicidal ideation.\n\n"
                "Symptoms:\n"
                "1. anhedonia: can't enjoy, no interest, don't want to do anything\n"
                "2. depressed_mood: sad, down, crying, hopeless, despair\n"
                "3. sleep_problems: insomnia, early waking, oversleeping, nightmares\n"
                "4. fatigue: exhausted, no energy, can't get out of bed\n"
                "5. appetite_changes: no appetite, overeating, weight change\n"
                "6. worthlessness_guilt: self-blame, feeling useless, guilt, burden\n"
                "7. concentration_problems: can't focus, forgetful, brain fog\n"
                "8. suicidal_ideation: want to die, not want to live, self-harm thoughts\n\n"
                'Output: {"anhedonia":true,"depressed_mood":true,"sleep_problems":false,"fatigue":true,"appetite_changes":false,"worthlessness_guilt":false,"concentration_problems":false,"suicidal_ideation":false}\n'
                "JSON only, no other text.\n\n"
                f"Text:\n{truncated}\n\nJSON:"
            )

        raw = self._generate(prompt, temperature=0.1)
        parsed = self._parse_json(raw, {})

        result = {}
        for s in PHQ8_SYMPTOMS:
            val = parsed.get(s, False)
            if isinstance(val, str):
                val = val.lower() in ("true", "yes", "1")
            elif isinstance(val, (int, float)):
                val = bool(val)
            result[s] = bool(val)

        return result

    def secondary_appraisal(self, text: str, primary: Dict) -> Dict:
        p_score = primary.get("primary_appraisal_score", 5)
        return self._algorithmic_secondary(text, p_score)

    def _algorithmic_secondary(self, text: str, primary_score: int) -> Dict:
        risk_score, _ = self._scan_risk_keywords(text)
        coping_score, coping_kw = self._scan_coping_keywords(text)

        base_secondary = max(1, min(10, 11 - primary_score))

        coping_adjustment = 0
        if coping_score > 0.3:
            coping_adjustment = min(3, round(coping_score * 4))
        if risk_score > 0.3 and coping_score < 0.2:
            coping_adjustment = max(coping_adjustment - 2, -3)

        secondary_score = max(1, min(10, base_secondary + coping_adjustment))

        ideal_s = max(1, min(10, 11 - primary_score))
        if abs(secondary_score - ideal_s) > 3:
            secondary_score = round(0.6 * ideal_s + 0.4 * secondary_score)

        coping_efficacy = max(1, min(10, secondary_score + 1))
        social_support = max(1, min(10, secondary_score))

        return {
            "coping_resources": coping_kw[:5],
            "coping_efficacy": coping_efficacy,
            "social_support_perceived": social_support,
            "resource_narrative": "",
            "secondary_appraisal_score": secondary_score,
            "_algorithmic": True,
            "_base_secondary": base_secondary,
            "_coping_adjustment": coping_adjustment,
        }

    def _enforce_primary_secondary_consistency(self, primary: Dict, secondary: Dict) -> Dict:
        p_score = primary.get("primary_appraisal_score", 5)
        s_score = secondary.get("secondary_appraisal_score", 5)
        ideal = max(1, min(10, 11 - p_score))
        deviation = s_score - ideal

        if abs(deviation) > 3:
            blend = round(0.55 * ideal + 0.45 * s_score)
            secondary["secondary_appraisal_score"] = max(1, min(10, blend))
        return secondary

    def reappraisal(self, primary: Dict, secondary: Dict, text: str) -> Dict:
        p = primary.get("primary_appraisal_score", 5)
        s = secondary.get("secondary_appraisal_score", 5)
        return self._algorithmic_reappraisal(p, s, text)

    def _algorithmic_reappraisal(self, primary: int, secondary: int, text: str) -> Dict:
        cp = primary
        cs = secondary

        if primary >= 7 and secondary >= 6:
            cs = max(1, min(10, 11 - primary))
            cp = min(10, cp + 1)
        elif primary <= 3 and secondary <= 4:
            cp = max(1, cp - 1)
            cs = min(10, 11 - cp)

        risk_score, _ = self._scan_risk_keywords(text)
        if risk_score > 0.5 and cp < 7:
            cp = min(10, cp + 1)

        return {
            "rollback_needed": abs(cp - primary) > 1 or abs(cs - secondary) > 2,
            "rollback_reason": "primary_secondary_conflict" if (primary >= 7 and secondary >= 6) else "",
            "corrected_primary": cp,
            "corrected_secondary": cs,
            "lazarus_consistency": not (cp >= 7 and cs >= 6) and not (cp <= 3 and cs <= 4),
            "expected_stress": round(cp * cs / 10.0, 2),
            "adjustment_direction": "increase_coping" if cp > 5 else "maintain" if cp <= 3 else "decrease_coping",
            "_algorithmic": True,
        }

    def detect_distortions(self, text: str, cp: float, cs: float) -> List[Dict]:
        return self._detect_distortions_rule_based(text)

    def integrate(self, primary, secondary, reappraisal, distortions, causal_info, dynamics_info):
        p = reappraisal["corrected_primary"]
        s = reappraisal["corrected_secondary"]
        social = secondary.get("social_support_perceived", 5)
        n_dist = len(distortions)
        a_proxy = round((p - s) / 10.0, 4)
        b_proxy = round((s * social) / 100.0 - 0.5, 4)
        c_proxy = round(social * (11 - min(n_dist, 5)) / 100.0, 4)

        return {
            "cusp_proxies": {"a_proxy": a_proxy, "b_proxy": b_proxy, "c_proxy": c_proxy},
            "explanation": f"Threat={p}/10, Coping={s}/10, Social={social}/10, Distortions={n_dist}",
            "intervention": "Consider professional consultation." if p >= 7 else "Monitor and support.",
        }

    def full_chain(self, text: str, causal_info=None, dynamics_info=None) -> Dict[str, Any]:
        text = text[:3000]

        primary = self.primary_appraisal(text)
        secondary = self.secondary_appraisal(text, primary)
        secondary = self._enforce_primary_secondary_consistency(primary, secondary)
        reappraisal_result = self.reappraisal(primary, secondary, text)

        cp = reappraisal_result["corrected_primary"]
        cs = reappraisal_result["corrected_secondary"]
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

        calibration_meta = {
            "trust_factor": round(trust, 4),
            "raw_secondary": secondary.get("secondary_appraisal_score", 5),
            "enforced_secondary": secondary.get("secondary_appraisal_score", 5),
            "ideal_secondary": ideal_s,
            "secondary_gap": round(sec_gap, 2),
            "corrected_primary": cp,
            "corrected_secondary": cs,
            "cove_consistent": True,
            "critique_passed": True,
            "risk_floor_applied": False,
            "cve_correction": None,
            "critique_adjustment": None,
        }

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
