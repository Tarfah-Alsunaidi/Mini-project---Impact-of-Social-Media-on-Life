import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

NUM = ["Age", "Daily_Usage_Hours", "Weekend_Extra_Hours", "Sleep_Duration_Hours",
       "Sleep_Quality_Score", "Perceived_Stress_Score", "Mental_Health_Index",
       "Academic_Performance_GPA", "Late_Night_Usage"]
CAT = ["Gender", "Academic_Level", "Primary_Platform", "Device_Type",
       "Social_Comparison_Frequency"]
CLASSES = ["Beneficial", "Negative", "Neutral"]
KNOWN_CATS = {
    "Gender": ["Female", "Male", "Non-Binary", "Prefer not to say"],
    "Academic_Level": ["High School", "Postgraduate", "Undergraduate"],
    "Primary_Platform": ["Instagram", "LinkedIn", "Reddit", "Snapchat", "TikTok", "X (Twitter)", "YouTube"],
    "Device_Type": ["Laptop/PC", "Smartphone", "Tablet"],
    "Social_Comparison_Frequency": ["Always", "Frequently", "Never", "Rarely", "Sometimes"],
}


def _dense(a):
    a = a.toarray() if hasattr(a, "toarray") else a
    return np.asarray(a, dtype=float)


def _split(model):
    if hasattr(model, "steps"):
        return (lambda raw: model[:-1].transform(raw)), model[-1]
    return None, model


def _names(labels, class_names):
    labels = np.asarray(labels)
    if class_names is not None and labels.dtype.kind in "iu":
        return np.asarray(class_names)[labels].astype(str)
    return labels.astype(str)


def export_for_story(svm, X_test_raw, y_test, prepare=None, class_names=None,
                     rival=None, rival_name="Rival model", rival_prepare=None,
                     out="model.json", tol=1e-6):
    pipe_prep, svc = _split(svm)
    prepare = prepare or pipe_prep
    if prepare is None:
        raise ValueError("Your SVM is not a Pipeline, so pass prepare=... (a function from raw columns to model input).")
    if not hasattr(svc, "support_vectors_"):
        raise ValueError(f"The last step is {type(svc).__name__}, not a fitted sklearn SVC.")
    if svc.kernel not in ("rbf", "linear", "poly", "sigmoid"):
        raise ValueError(f"Kernel '{svc.kernel}' can't run in the browser. Use rbf, linear, poly or sigmoid.")

    raw = X_test_raw.copy()
    missing = [c for c in NUM + CAT if c not in raw.columns]
    if missing:
        raise ValueError(f"X_test_raw needs the original columns. Missing: {missing}. "
                         "Tip: df.loc[X_test.index] gives the raw rows of your test split.")
    bool_like = raw["Late_Night_Usage"].dtype == bool
    medians = {c: float(pd.to_numeric(raw[c], errors="coerce").median()) for c in NUM}
    filled = raw.copy()
    for c in NUM:
        if c == "Late_Night_Usage":
            continue
        filled[c] = pd.to_numeric(filled[c], errors="coerce").fillna(medians[c])
    for c in CAT:
        filled[c] = filled[c].fillna(filled[c].mode()[0])

    ref = filled.iloc[[0]].copy()
    if bool_like:
        ref["Late_Night_Usage"] = False
    else:
        ref["Late_Night_Usage"] = 0
    probes, tags = [ref], [("ref", None)]
    for c in NUM:
        p = ref.copy()
        p[c] = True if (c == "Late_Night_Usage" and bool_like) else (1 if c == "Late_Night_Usage" else float(ref[c].iloc[0]) + 1.0)
        probes.append(p); tags.append(("num", c))
    cats = {c: sorted(set(KNOWN_CATS[c]) | set(filled[c].astype(str))) for c in CAT}
    for c in CAT:
        for v in cats[c]:
            p = ref.copy(); p[c] = v
            probes.append(p); tags.append(("cat", (c, v)))
    Z = _dense(prepare(pd.concat(probes, ignore_index=True)))
    z0 = Z[0]
    refv = {c: (0.0 if c == "Late_Night_Usage" else float(ref[c].iloc[0])) for c in NUM}
    A, D = {}, {c: {} for c in CAT}
    for (kind, key), row in zip(tags[1:], Z[1:]):
        if kind == "num":
            A[key] = row - z0
        else:
            D[key[0]][key[1]] = row - z0

    def rebuild(df):
        z = np.tile(z0, (len(df), 1))
        for c in NUM:
            x = df[c].astype(float).to_numpy() if c != "Late_Night_Usage" else df[c].astype(bool).astype(float).to_numpy()
            z += np.outer(x - refv[c], A[c])
        for c in CAT:
            z += np.vstack([D[c].get(str(v), np.zeros_like(z0)) for v in df[c]])
        return z

    real = _dense(prepare(filled))
    err = np.abs(rebuild(filled) - real).max(axis=0)
    if err.max() > tol:
        bad = []
        for c in NUM:
            if c == "Late_Night_Usage":
                continue
            p = ref.copy(); p[c] = refv[c] + 3.0
            if np.abs(_dense(prepare(p))[0] - (z0 + 3 * A[c])).max() > tol:
                bad.append(c)
        raise ValueError("Your preprocessing isn't a plain scale/encode step, so the browser can't copy it exactly. "
                         f"Columns that behave non-linearly: {bad or 'a combination of columns'}. "
                         "Ask Claude to add that transform to the page.")

    svm_pred = _names(svc.predict(real), class_names)
    truth = _names(y_test, class_names) if np.asarray(y_test).dtype.kind in "iu" else np.asarray(y_test).astype(str)
    if set(np.unique(truth)) - set(CLASSES):
        raise ValueError(f"Expected classes {CLASSES}, got {sorted(set(truth))}. "
                         "If your labels are numbers, pass class_names=le.classes_.")
    svm_classes = list(_names(svc.classes_, class_names))

    rival_pred = None
    if rival is not None:
        if hasattr(rival, "steps"):
            rival_pred = rival.predict(raw)
        else:
            rival_pred = rival.predict(_dense((rival_prepare or prepare)(filled)))
        rival_pred = _names(rival_pred, class_names)

    def metrics(p):
        return {"accuracy": round(float(accuracy_score(truth, p)), 6),
                "macro_f1": round(float(f1_score(truth, p, average="macro")), 6),
                "weighted_f1": round(float(f1_score(truth, p, average="weighted")), 6),
                "confusion": confusion_matrix(truth, p, labels=svm_classes).tolist()}

    test = filled[[c for c in (["Student_ID"] if "Student_ID" in filled else []) + NUM + CAT]].copy()
    test["Late_Night_Usage"] = test["Late_Night_Usage"].astype(bool).astype(int) if test["Late_Night_Usage"].dtype != float else test["Late_Night_Usage"]
    if "Student_ID" not in test:
        test.insert(0, "Student_ID", [f"STU_{i}" for i in range(len(test))])
    test["Overall_Impact"] = truth
    if rival_pred is not None:
        test["_rival"] = rival_pred
    test["_svm"] = svm_pred

    gamma = float(getattr(svc, "_gamma", svc.gamma if isinstance(svc.gamma, float) else 0.0))
    model = {
        "format": 2,
        "classes": svm_classes, "num": NUM, "cat": CAT,
        "medians": [medians[c] for c in NUM],
        "categories": [cats[c] for c in CAT],
        "prep": {"z0": np.round(z0, 10).tolist(), "ref": [refv[c] for c in NUM],
                 "A": [np.round(A[c], 10).tolist() for c in NUM],
                 "D": [{v: np.round(d, 10).tolist() for v, d in D[c].items()} for c in CAT]},
        "svm": {"kernel": svc.kernel, "gamma": gamma, "coef0": float(svc.coef0), "degree": int(svc.degree),
                "sv": np.round(svc.support_vectors_ if not hasattr(svc.support_vectors_, "toarray") else svc.support_vectors_.toarray(), 8).tolist(),
                "n_support": svc.n_support_.tolist(), "dual_coef": np.round(_dense(svc.dual_coef_), 10).tolist(),
                "intercept": svc.intercept_.tolist(),
                "params": {"C": str(svc.C), "gamma": str(svc.gamma), "kernel": svc.kernel,
                           "class_weight": str(svc.class_weight)}},
        "rival_name": rival_name if rival is not None else None,
        "metrics": {"SVM": metrics(svm_pred), **({"Rival": metrics(rival_pred)} if rival_pred is not None else {})},
        "test": test.replace({np.nan: None}).to_dict(orient="records"),
    }
    with open(out, "w") as f:
        json.dump(model, f, separators=(",", ":"))
    m = model["metrics"]["SVM"]
    print(f"Saved {out}: {len(test)} test students, {len(model['svm']['sv'])} support vectors, "
          f"SVM accuracy {m['accuracy']:.3f}, weighted F1 {m['weighted_f1']:.3f}, macro F1 {m['macro_f1']:.3f}"
          + (f" | {rival_name} accuracy {model['metrics']['Rival']['accuracy']:.3f}" if rival_pred is not None else ""))
    print("Preprocessing check: passed on every test row.")
    return out
