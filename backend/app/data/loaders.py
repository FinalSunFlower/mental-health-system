import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

_RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")


class SachsLoader:
    SACHS_VAR_NAMES = [
        "PKC",
        "PKA",
        "Raf",
        "Mek",
        "Erk",
        "Akt",
        "Jnk",
        "P38",
        "Plcg",
        "PIP3",
        "PIP2",
    ]

    SACHS_ADJ_TRUE = np.array(
        [
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        ],
        dtype=np.float64,
    )

    def load(self) -> Tuple[np.ndarray, np.ndarray, list]:
        data_file = os.path.join(_RAW_DIR, "sachs_data.csv")
        if not os.path.exists(data_file):
            raise FileNotFoundError(
                f"Sachs dataset not found at {data_file}. "
                "Please download from https://www.bnlearn.com/book-crc/code/sachs.data.txt "
                "and place as sachs_data.csv in app/data/raw/"
            )
        try:
            df = pd.read_csv(data_file, sep=None, engine="python")
        except Exception as e:
            raise ValueError(f"Failed to parse Sachs dataset: {e}")
        df = df.dropna()
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna()
        if df.shape[1] != 11:
            df = df.iloc[:, :11]
        X = df.values.astype(np.float64)
        adj_file = os.path.join(_RAW_DIR, "sachs_adj_true.csv")
        if os.path.exists(adj_file):
            try:
                adj_df = pd.read_csv(adj_file, header=None)
                adj_true = adj_df.values.astype(np.float64)
            except Exception:
                adj_true = self.SACHS_ADJ_TRUE.copy()
        else:
            adj_true = self.SACHS_ADJ_TRUE.copy()
        var_names = self.SACHS_VAR_NAMES[: X.shape[1]]
        return X, adj_true, var_names


class NHANESLoader:
    DPQ_COLUMNS = [
        "DPQ010",
        "DPQ020",
        "DPQ030",
        "DPQ040",
        "DPQ050",
        "DPQ060",
        "DPQ070",
        "DPQ080",
        "DPQ090",
        "DPQ100",
    ]

    PHQ9_LABELS = [
        "little_interest",
        "depressed",
        "sleep_problems",
        "low_energy",
        "appetite_issues",
        "self_worth",
        "concentration",
        "psychomotor",
        "suicidal_ideation",
        "functional_impairment",
    ]

    def load(self) -> Tuple[np.ndarray, list]:
        data_file = os.path.join(_RAW_DIR, "nhanes_dpq.csv")
        if not os.path.exists(data_file):
            xpt_file = os.path.join(_RAW_DIR, "DPQ_J.XPT")
            if os.path.exists(xpt_file):
                try:
                    df = pd.read_sas(xpt_file, format="xport")
                    df.to_csv(data_file, index=False)
                except Exception as e:
                    raise FileNotFoundError(
                        f"Failed to convert NHANES XPT to CSV: {e}. "
                        "Please provide nhanes_dpq.csv in app/data/raw/"
                    )
            else:
                raise FileNotFoundError(
                    f"NHANES dataset not found at {data_file} or {xpt_file}. "
                    "Please download from https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DPQ_J.XPT "
                    "and place in app/data/raw/"
                )
        try:
            df = pd.read_csv(data_file)
        except Exception as e:
            raise ValueError(f"Failed to parse NHANES dataset: {e}")
        dpq_cols = [c for c in df.columns if c.startswith("DPQ")]
        if not dpq_cols:
            dpq_cols = [c for c in df.columns if "dpq" in c.lower()]
        if not dpq_cols:
            dpq_cols = self.DPQ_COLUMNS[:9]
        df_dpq = df[dpq_cols].copy()
        df_dpq = df_dpq.apply(pd.to_numeric, errors="coerce")
        df_dpq = df_dpq.replace([7, 9], np.nan)
        df_dpq = df_dpq.where(df_dpq > 1e-10, 0.0)
        df_dpq = df_dpq.dropna(thresh=max(1, len(dpq_cols) // 2))
        col_means = df_dpq.mean()
        df_dpq = df_dpq.fillna(col_means)
        X = df_dpq.values.astype(np.float64)
        var_names = self.PHQ9_LABELS[: X.shape[1]]
        return X, var_names


class KossakowskiLoader:
    EXPECTED_COLUMNS = ["stress", "resilience", "social_support", "mood"]

    def load(self) -> pd.DataFrame:
        data_file = os.path.join(_RAW_DIR, "kossakowski_esm.csv")
        if not os.path.exists(data_file):
            json_file = os.path.join(_RAW_DIR, "kossakowski_esm.json")
            if os.path.exists(json_file):
                try:
                    df = pd.read_json(json_file)
                    df.to_csv(data_file, index=False)
                except Exception as e:
                    raise FileNotFoundError(
                        f"Failed to convert Kossakowski JSON to CSV: {e}. "
                        "Please provide kossakowski_esm.csv in app/data/raw/"
                    )
            else:
                raise FileNotFoundError(
                    f"Kossakowski dataset not found at {data_file}. "
                    "Please provide 239-day ESM data as kossakowski_esm.csv "
                    "with columns: stress, resilience, social_support, mood"
                )
        try:
            df = pd.read_csv(data_file)
        except Exception as e:
            raise ValueError(f"Failed to parse Kossakowski dataset: {e}")
        col_mapping = {}
        for col in df.columns:
            col_lower = col.lower().strip()
            if "stress" in col_lower:
                col_mapping[col] = "stress"
            elif "resil" in col_lower:
                col_mapping[col] = "resilience"
            elif "social" in col_lower or "support" in col_lower:
                col_mapping[col] = "social_support"
            elif "mood" in col_lower or "affect" in col_lower:
                col_mapping[col] = "mood"
        df = df.rename(columns=col_mapping)
        for target in self.EXPECTED_COLUMNS:
            if target not in df.columns:
                if target == "mood" and "affect" in df.columns:
                    df["mood"] = df["affect"]
                else:
                    df[target] = np.nan
        df = df[self.EXPECTED_COLUMNS].copy()
        for col in self.EXPECTED_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.fillna(df.mean())
        df = df.dropna(how="all")
        return df


class DAICWOZLoader:
    def load(self) -> Dict:
        data_dir = os.path.join(_RAW_DIR, "daic_woz")
        transcripts: Dict[str, str] = {}
        labels: Dict[str, int] = {}
        if os.path.isdir(data_dir):
            for fname in sorted(os.listdir(data_dir)):
                fpath = os.path.join(data_dir, fname)
                if fname.endswith("_TRANSCRIPT.csv") or fname.endswith("_transcript.csv"):
                    pid = fname.split("_")[0]
                    try:
                        tdf = pd.read_csv(fpath, sep="\t", engine="python")
                        text_parts = []
                        for c in tdf.columns:
                            if "value" in c.lower() or "text" in c.lower():
                                vals = tdf[c].dropna().astype(str).tolist()
                                text_parts.extend(vals)
                        if not text_parts:
                            text_parts = tdf.iloc[:, -1].dropna().astype(str).tolist()
                        transcripts[pid] = " ".join(text_parts)
                    except Exception:
                        continue

        if not transcripts:
            csv_file = os.path.join(_RAW_DIR, "daic_woz.csv")
            if os.path.exists(csv_file):
                try:
                    df = pd.read_csv(csv_file)
                except Exception as e:
                    raise ValueError(f"Failed to parse DAIC-WOZ CSV: {e}")
                id_col = None
                for c in df.columns:
                    if "id" in c.lower() or "participant" in c.lower():
                        id_col = c
                        break
                if id_col is None:
                    id_col = df.columns[0]
                text_col = None
                for c in df.columns:
                    if "transcript" in c.lower() or "text" in c.lower() or "response" in c.lower():
                        text_col = c
                        break
                label_col = None
                for c in df.columns:
                    if "phq" in c.lower() or "score" in c.lower() or "label" in c.lower() or "depression" in c.lower():
                        label_col = c
                        break
                for _, row in df.iterrows():
                    pid = str(row[id_col])
                    if text_col and text_col in row:
                        transcripts[pid] = str(row[text_col]) if pd.notna(row[text_col]) else ""
                    if label_col and label_col in row:
                        try:
                            labels[pid] = int(float(row[label_col]))
                        except (ValueError, TypeError):
                            labels[pid] = 0
                return {"transcripts": transcripts, "labels": labels}
            raise FileNotFoundError(
                f"DAIC-WOZ dataset not found at {data_dir} or {csv_file}. "
                "Please download from https://dcapswoz.ict.usc.edu/ "
                "and place transcript files in app/data/raw/daic_woz/ "
                "or a combined CSV as daic_woz.csv"
            )
        label_file = os.path.join(data_dir, "Labels.csv")
        if not os.path.exists(label_file):
            for alt in ["labels.csv", "scores.csv", "PHQ8_labels.csv"]:
                alt_path = os.path.join(data_dir, alt)
                if os.path.exists(alt_path):
                    label_file = alt_path
                    break
        if os.path.exists(label_file):
            try:
                ldf = pd.read_csv(label_file)
                id_col = ldf.columns[0]
                phq_col = None
                for c in ldf.columns:
                    if "phq" in c.lower() or "score" in c.lower():
                        phq_col = c
                        break
                if phq_col is None:
                    phq_col = ldf.columns[-1]
                for _, row in ldf.iterrows():
                    pid = str(int(row[id_col])) if isinstance(row[id_col], (int, float)) else str(row[id_col])
                    try:
                        labels[pid] = int(float(row[phq_col]))
                    except (ValueError, TypeError):
                        labels[pid] = 0
            except Exception:
                pass
        return {"transcripts": transcripts, "labels": labels}


class StudentLifeLoader:
    def load(self) -> Dict:
        data_dir = os.path.join(_RAW_DIR, "studentlife")
        csv_file = os.path.join(_RAW_DIR, "studentlife.csv")
        if os.path.isdir(data_dir):
            return self._load_from_dir(data_dir)
        if os.path.exists(csv_file):
            return self._load_from_csv(csv_file)
        raise FileNotFoundError(
            f"StudentLife dataset not found at {data_dir} or {csv_file}. "
            "Please download from https://studentlife.cs.dartmouth.edu/ "
            "and place data in app/data/raw/studentlife/ "
            "or a combined CSV as studentlife.csv"
        )

    def _load_from_csv(self, csv_file: str) -> Dict:
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            raise ValueError(f"Failed to parse StudentLife CSV: {e}")
        phq9_series = {}
        stress_series = {}
        sensor_data = {}
        id_col = None
        for c in df.columns:
            if "id" in c.lower() or "student" in c.lower() or "participant" in c.lower():
                id_col = c
                break
        if id_col is None:
            id_col = df.columns[0]
        phq_cols = [c for c in df.columns if "phq" in c.lower()]
        stress_cols = [c for c in df.columns if "stress" in c.lower() or "pss" in c.lower()]
        sensor_cols = [c for c in df.columns if c not in [id_col] + phq_cols + stress_cols]
        for pid, group in df.groupby(id_col):
            pid_str = str(pid)
            if phq_cols:
                phq9_series[pid_str] = group[phq_cols].apply(pd.to_numeric, errors="coerce").values.tolist()
            if stress_cols:
                stress_series[pid_str] = group[stress_cols].apply(pd.to_numeric, errors="coerce").values.tolist()
            if sensor_cols:
                sensor_data[pid_str] = group[sensor_cols].apply(pd.to_numeric, errors="coerce").values.tolist()
        return {
            "phq9_series": phq9_series,
            "stress_series": stress_series,
            "sensor_data": sensor_data,
        }

    def _load_from_dir(self, data_dir: str) -> Dict:
        phq9_series: Dict[str, list] = {}
        stress_series: Dict[str, list] = {}
        sensor_data: Dict[str, dict] = {}
        phq_dir = os.path.join(data_dir, "phq9")
        if os.path.isdir(phq_dir):
            for fname in sorted(os.listdir(phq_dir)):
                if fname.endswith(".csv"):
                    pid = fname.replace(".csv", "").replace("phq9_", "").replace("student_", "")
                    fpath = os.path.join(phq_dir, fname)
                    try:
                        df = pd.read_csv(fpath)
                        df = df.apply(pd.to_numeric, errors="coerce")
                        df = df.fillna(df.mean())
                        phq9_series[pid] = df.values.tolist()
                    except Exception:
                        continue
        stress_dir = os.path.join(data_dir, "stress")
        if os.path.isdir(stress_dir):
            for fname in sorted(os.listdir(stress_dir)):
                if fname.endswith(".csv"):
                    pid = fname.replace(".csv", "").replace("stress_", "").replace("student_", "")
                    fpath = os.path.join(stress_dir, fname)
                    try:
                        df = pd.read_csv(fpath)
                        df = df.apply(pd.to_numeric, errors="coerce")
                        df = df.fillna(df.mean())
                        stress_series[pid] = df.values.tolist()
                    except Exception:
                        continue
        sensor_dir = os.path.join(data_dir, "sensing")
        if os.path.isdir(sensor_dir):
            for fname in sorted(os.listdir(sensor_dir)):
                if fname.endswith(".csv"):
                    sensor_type = fname.replace(".csv", "")
                    fpath = os.path.join(sensor_dir, fname)
                    try:
                        df = pd.read_csv(fpath)
                        df = df.apply(pd.to_numeric, errors="coerce")
                        df = df.fillna(df.mean())
                        sensor_data[sensor_type] = df.values.tolist()
                    except Exception:
                        continue
        return {
            "phq9_series": phq9_series,
            "stress_series": stress_series,
            "sensor_data": sensor_data,
        }
