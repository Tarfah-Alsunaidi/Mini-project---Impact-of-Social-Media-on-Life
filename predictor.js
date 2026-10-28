

function makePredictor(M) {
  function transform(row) {
    const val = (k, i) => {
      let v = row[k];
      if (typeof v === "boolean") v = v ? 1 : 0;
      if (v === null || v === undefined || v === "" || Number.isNaN(+v)) v = M.medians[i];
      return +v;
    };
    if (M.prep) {
      const z = M.prep.z0.slice();
      M.num.forEach((k, i) => {
        const d = val(k, i) - M.prep.ref[i], a = M.prep.A[i];
        if (d) for (let j = 0; j < z.length; j++) z[j] += a[j] * d;
      });
      M.cat.forEach((k, c) => {
        const delta = M.prep.D[c][String(row[k])];
        if (delta) for (let j = 0; j < z.length; j++) z[j] += delta[j];
      });
      return z;
    }
    const z = [];
    M.num.forEach((k, i) => z.push((val(k, i) - M.mean[i]) / M.scale[i]));
    M.cat.forEach((k, j) => M.categories[j].forEach(c => z.push(String(row[k]) === c ? 1 : 0)));
    return z;
  }
  const S = M.svm, K = M.classes.length;
  const starts = [0];
  S.n_support.forEach((n, i) => starts.push(starts[i] + n));
  function kern(a, b) {
    let s = 0;
    if (S.kernel === "rbf") { for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; } return Math.exp(-S.gamma * s); }
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    if (S.kernel === "linear") return s;
    if (S.kernel === "poly") return Math.pow(S.gamma * s + (S.coef0 || 0), S.degree || 3);
    if (S.kernel === "sigmoid") return Math.tanh(S.gamma * s + (S.coef0 || 0));
    return s;
  }
  function svm(row) {
    const z = transform(row);
    const k = S.sv.map(sv => kern(z, sv));
    const votes = new Array(K).fill(0), pairs = [];
    let p = 0;
    for (let i = 0; i < K; i++) for (let j = i + 1; j < K; j++) {
      let d = S.intercept[p];
      for (let s = starts[i]; s < starts[i + 1]; s++) d += S.dual_coef[j - 1][s] * k[s];
      for (let s = starts[j]; s < starts[j + 1]; s++) d += S.dual_coef[i][s] * k[s];
      if (d > 0) votes[i]++; else votes[j]++;
      pairs.push({ i, j, d });
      p++;
    }
    let w = 0; votes.forEach((v, c) => { if (v > votes[w]) w = c; });
    let margin = Infinity;
    pairs.forEach(({ i, j, d }) => {
      if (i === w) margin = Math.min(margin, d);
      if (j === w) margin = Math.min(margin, -d);
    });
    return { label: M.classes[w], votes, margin, z };
  }
  function lr(row, zIn) {
    if (!M.lr) return null;
    const z = zIn || transform(row);
    const logits = M.lr.coef.map((c, k) => c.reduce((s, v, i) => s + v * z[i], M.lr.intercept[k]));
    const mx = Math.max(...logits), e = logits.map(l => Math.exp(l - mx)), t = e.reduce((a, b) => a + b, 0);
    const prob = e.map(v => v / t);
    let w = 0; prob.forEach((v, c) => { if (v > prob[w]) w = c; });
    return { label: M.classes[w], prob };
  }
  function project(z) {
    const c = z.map((v, i) => v - M.pca.mean[i]);
    return M.pca.components.map(row => row.reduce((s, v, i) => s + v * c[i], 0));
  }
  return { transform, svm, lr, project };
}
if (typeof module !== "undefined") module.exports = { makePredictor };
