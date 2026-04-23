from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import joblib
import os
import numpy as np

app = Flask(__name__)

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# Chargement des modèles
try:
    imputer = joblib.load(os.path.join(BASE, "models/imputer.joblib"))
    scaler  = joblib.load(os.path.join(BASE, "models/scaler.joblib"))
    modele  = joblib.load(os.path.join(BASE, "models/random_forest.joblib"))
    feature_names = scaler.feature_names_in_
    MODELS_LOADED = True
except Exception as e:
    print(f"[WARN] Modèles non chargés : {e}")
    MODELS_LOADED = False
    feature_names = []

HTML = """
<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Dashboard Prédiction Churn</title>
  <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --navy:   #0F1F3D;
      --teal:   #0D9488;
      --teal2:  #14B8A6;
      --teal-light: #E8F4F2;
      --white:  #FFFFFF;
      --light:  #F5F9F8;
      --gray:   #64748B;
      --gray2:  #94A3B8;
      --dark:   #1E293B;
      --accent: #F59E0B;
      --red:    #EF4444;
      --green:  #10B981;
      --border: #E2EBE9;
      --shadow: 0 4px 24px rgba(15,31,61,0.08);
      --shadow-lg: 0 12px 40px rgba(15,31,61,0.14);
    }

    body {
      font-family: 'DM Sans', sans-serif;
      background: var(--light);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 40px 20px;
      color: var(--dark);
    }

    /* Fond décoratif */
    body::before {
      content: '';
      position: fixed;
      top: -200px; right: -200px;
      width: 600px; height: 600px;
      background: radial-gradient(circle, rgba(13,148,136,0.07) 0%, transparent 70%);
      pointer-events: none;
      z-index: 0;
    }

    .page-wrapper {
      width: 100%;
      max-width: 560px;
      position: relative;
      z-index: 1;
    }

    /* Header */
    .header {
      text-align: center;
      margin-bottom: 32px;
    }

    .header-icon {
      font-size: 36px;
      margin-bottom: 8px;
      display: block;
      filter: drop-shadow(0 2px 8px rgba(13,148,136,0.3));
    }

    .header h1 {
      font-size: 28px;
      font-weight: 700;
      color: var(--navy);
      letter-spacing: -0.5px;
    }

    .header p {
      margin-top: 6px;
      font-size: 14px;
      color: var(--gray);
      font-weight: 400;
    }

    /* Card principale */
    .card {
      background: var(--white);
      border-radius: 20px;
      padding: 32px;
      box-shadow: var(--shadow-lg);
      border: 1px solid var(--border);
    }

    /* Groupe de champs */
    .fields-section {
      margin-bottom: 24px;
    }

    .section-label {
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 1.2px;
      text-transform: uppercase;
      color: var(--teal);
      margin-bottom: 16px;
    }

    .field-group {
      margin-bottom: 18px;
    }

    .field-group label {
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 13px;
      font-weight: 500;
      color: var(--dark);
      margin-bottom: 7px;
    }

    .field-icon {
      width: 20px;
      height: 20px;
      background: var(--teal-light);
      border-radius: 5px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-size: 11px;
    }

    .field-hint {
      font-size: 11px;
      color: var(--gray2);
      font-weight: 400;
      margin-left: auto;
    }

    input[type="number"] {
      width: 100%;
      padding: 12px 16px;
      border: 1.5px solid var(--border);
      border-radius: 10px;
      font-family: 'DM Mono', monospace;
      font-size: 14px;
      color: var(--dark);
      background: var(--light);
      transition: all 0.2s ease;
      outline: none;
      -moz-appearance: textfield;
    }

    input[type="number"]::-webkit-outer-spin-button,
    input[type="number"]::-webkit-inner-spin-button { -webkit-appearance: none; }

    input[type="number"]:focus {
      border-color: var(--teal);
      background: var(--white);
      box-shadow: 0 0 0 3px rgba(13,148,136,0.1);
    }

    input[type="number"].has-value {
      border-color: var(--teal2);
      background: var(--white);
    }

    /* Divider */
    .divider {
      height: 1px;
      background: var(--border);
      margin: 24px 0;
    }

    /* Bouton */
    .btn-predict {
      width: 100%;
      padding: 14px;
      background: var(--navy);
      color: var(--white);
      border: none;
      border-radius: 12px;
      font-family: 'DM Sans', sans-serif;
      font-size: 15px;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      transition: all 0.2s ease;
      letter-spacing: 0.2px;
    }

    .btn-predict:hover {
      background: var(--teal);
      transform: translateY(-1px);
      box-shadow: 0 6px 20px rgba(13,148,136,0.3);
    }

    .btn-predict:active { transform: translateY(0); }

    .btn-predict.loading {
      background: var(--gray);
      pointer-events: none;
    }

    .spinner {
      width: 16px; height: 16px;
      border: 2px solid rgba(255,255,255,0.3);
      border-top-color: white;
      border-radius: 50%;
      animation: spin 0.7s linear infinite;
      display: none;
    }
    .btn-predict.loading .spinner { display: block; }
    .btn-predict.loading .btn-text { display: none; }

    @keyframes spin { to { transform: rotate(360deg); } }

    /* Résultat */
    .result-box {
      margin-top: 24px;
      border-radius: 14px;
      padding: 24px;
      display: none;
      animation: fadeSlideUp 0.4s ease forwards;
    }

    @keyframes fadeSlideUp {
      from { opacity: 0; transform: translateY(10px); }
      to   { opacity: 1; transform: translateY(0); }
    }

    .result-box.churn {
      background: linear-gradient(135deg, #FEF2F2, #FFF5F5);
      border: 1.5px solid #FCA5A5;
    }

    .result-box.no-churn {
      background: linear-gradient(135deg, #F0FDF9, #F0FFFE);
      border: 1.5px solid #6EE7B7;
    }

    .result-header {
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 16px;
    }

    .result-emoji {
      font-size: 32px;
      line-height: 1;
    }

    .result-title {
      font-size: 18px;
      font-weight: 700;
    }

    .result-box.churn .result-title { color: #DC2626; }
    .result-box.no-churn .result-title { color: #059669; }

    .result-subtitle {
      font-size: 13px;
      color: var(--gray);
      margin-top: 2px;
    }

    /* Barre de probabilité */
    .proba-section { margin-top: 4px; }

    .proba-label {
      display: flex;
      justify-content: space-between;
      font-size: 12px;
      font-weight: 500;
      color: var(--gray);
      margin-bottom: 8px;
    }

    .proba-value {
      font-family: 'DM Mono', monospace;
      font-size: 22px;
      font-weight: 500;
    }
    .result-box.churn .proba-value { color: #DC2626; }
    .result-box.no-churn .proba-value { color: #059669; }

    .proba-bar-bg {
      height: 8px;
      background: rgba(0,0,0,0.06);
      border-radius: 99px;
      overflow: hidden;
    }

    .proba-bar-fill {
      height: 100%;
      border-radius: 99px;
      transition: width 0.8s cubic-bezier(0.4,0,0.2,1);
      width: 0%;
    }
    .result-box.churn .proba-bar-fill { background: linear-gradient(90deg, #F87171, #DC2626); }
    .result-box.no-churn .proba-bar-fill { background: linear-gradient(90deg, #34D399, #059669); }

    /* Pills infos */
    .info-pills {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 14px;
    }

    .pill {
      background: rgba(255,255,255,0.7);
      border: 1px solid rgba(0,0,0,0.06);
      border-radius: 20px;
      padding: 5px 12px;
      font-size: 12px;
      color: var(--dark);
      font-weight: 500;
    }

    .pill span { color: var(--gray); font-weight: 400; }

    /* Erreur */
    .error-box {
      margin-top: 20px;
      background: #FEF2F2;
      border: 1.5px solid #FCA5A5;
      border-radius: 12px;
      padding: 14px 16px;
      font-size: 13px;
      color: #DC2626;
      display: none;
    }

    /* Footer */
    .footer {
      text-align: center;
      margin-top: 24px;
      font-size: 12px;
      color: var(--gray2);
    }

    .footer strong { color: var(--teal); }
  </style>
</head>
<body>

<div class="page-wrapper">

  <!-- Header -->
  <div class="header">
    <span class="header-icon">🛒</span>
    <h1>Dashboard Prédiction Churn</h1>
    <p>Entrez les données client pour obtenir une prédiction instantanée</p>
  </div>

  <!-- Card -->
  <div class="card">

    <div class="section-label">📊 Données RFM du client</div>

    <!-- Recency -->
    <div class="field-group">
      <label>
        <span class="field-icon">📅</span>
        Recency (jours)
        <span class="field-hint">0 – 400</span>
      </label>
      <input type="number" id="recency" placeholder="ex : 45" min="0" max="400">
    </div>

    <!-- Frequency -->
    <div class="field-group">
      <label>
        <span class="field-icon">🔁</span>
        Frequency (nb achats)
        <span class="field-hint">1 – 50</span>
      </label>
      <input type="number" id="frequency" placeholder="ex : 8" min="1" max="50">
    </div>

    <!-- MonetaryTotal -->
    <div class="field-group">
      <label>
        <span class="field-icon">💷</span>
        MonetaryTotal (£)
        <span class="field-hint">0 – 15000</span>
      </label>
      <input type="number" id="monetary" placeholder="ex : 1250" min="0" max="15000">
    </div>

    <div class="divider"></div>

    <!-- Bouton -->
    <button class="btn-predict" id="btn" onclick="predict()">
      <div class="spinner"></div>
      <span class="btn-text">🔍 Prédire</span>
    </button>

    <!-- Résultat -->
    <div class="result-box" id="result">
      <div class="result-header">
        <span class="result-emoji" id="result-emoji"></span>
        <div>
          <div class="result-title" id="result-title"></div>
          <div class="result-subtitle" id="result-subtitle"></div>
        </div>
      </div>

      <div class="proba-section">
        <div class="proba-label">
          <span>Probabilité de churn</span>
          <span class="proba-value" id="proba-value"></span>
        </div>
        <div class="proba-bar-bg">
          <div class="proba-bar-fill" id="proba-bar"></div>
        </div>
      </div>

      <div class="info-pills" id="pills"></div>
    </div>

    <!-- Erreur -->
    <div class="error-box" id="error-box"></div>

  </div>

  <div class="footer">
    Modèle <strong>Random Forest</strong> · Pipeline ML Retail · GI2 2025–2026
  </div>

</div>

<script>
  // Style input quand il a une valeur
  document.querySelectorAll('input[type="number"]').forEach(inp => {
    inp.addEventListener('input', () => {
      inp.classList.toggle('has-value', inp.value !== '');
    });
  });

  async function predict() {
    const recency  = document.getElementById('recency').value;
    const frequency= document.getElementById('frequency').value;
    const monetary = document.getElementById('monetary').value;

    // Validation
    const errorBox = document.getElementById('error-box');
    errorBox.style.display = 'none';

    if (recency === '' || frequency === '' || monetary === '') {
      errorBox.textContent = '⚠️ Veuillez remplir tous les champs avant de prédire.';
      errorBox.style.display = 'block';
      return;
    }

    // UI loading
    const btn = document.getElementById('btn');
    btn.classList.add('loading');
    document.getElementById('result').style.display = 'none';

    try {
      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          Recency:       parseFloat(recency),
          Frequency:     parseFloat(frequency),
          MonetaryTotal: parseFloat(monetary)
        })
      });

      const data = await res.json();

      if (data.erreur) {
        errorBox.textContent = '❌ Erreur : ' + data.erreur;
        errorBox.style.display = 'block';
        return;
      }

      showResult(data, parseFloat(recency), parseFloat(frequency), parseFloat(monetary));

    } catch(e) {
      errorBox.textContent = '❌ Impossible de contacter le serveur : ' + e.message;
      errorBox.style.display = 'block';
    } finally {
      btn.classList.remove('loading');
    }
  }

  function showResult(data, recency, frequency, monetary) {
    const isChurn = data.churn === 1;
    const proba   = data.probabilite_churn;
    const pct     = Math.round(proba * 100);

    const box = document.getElementById('result');
    box.className = 'result-box ' + (isChurn ? 'churn' : 'no-churn');

    document.getElementById('result-emoji').textContent  = isChurn ? '⚠️' : '✅';
    document.getElementById('result-title').textContent  = isChurn ? 'Client à risque de churn' : 'Client fidèle';
    document.getElementById('result-subtitle').textContent = isChurn
      ? 'Ce client risque de quitter la plateforme — action recommandée'
      : 'Ce client présente un faible risque de départ';

    document.getElementById('proba-value').textContent = pct + '%';

    box.style.display = 'block';

    // Animer la barre
    setTimeout(() => {
      document.getElementById('proba-bar').style.width = pct + '%';
    }, 100);

    // Pills
    const rfm = monetary / (frequency || 1);
    const segment = frequency >= 10 ? 'Champion' : frequency >= 5 ? 'Fidèle' : recency < 30 ? 'Potentiel' : 'Dormant';
    document.getElementById('pills').innerHTML = `
      <div class="pill">Recency <span>${recency}j</span></div>
      <div class="pill">Frequency <span>${frequency} achats</span></div>
      <div class="pill">Panier moy. <span>£${Math.round(rfm)}</span></div>
      <div class="pill">Segment <span>${segment}</span></div>
    `;
  }
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"erreur": "Aucune donnée reçue"}), 400

        if not MODELS_LOADED:
            # Mode démo si modèles pas encore entraînés
            import random
            recency  = float(data.get("Recency", 50))
            frequency= float(data.get("Frequency", 5))
            monetary = float(data.get("MonetaryTotal", 500))

            # Heuristique simple pour la démo
            score = (recency / 400) * 0.5 + (1 - min(frequency, 20) / 20) * 0.3 + (1 - min(monetary, 5000) / 5000) * 0.2
            proba = round(min(max(score + random.uniform(-0.05, 0.05), 0.01), 0.99), 3)
            churn = 1 if proba > 0.5 else 0
            return jsonify({
                "churn": churn,
                "probabilite_churn": proba,
                "interpretation": "Client à risque (démo)" if churn == 1 else "Client fidèle (démo)",
                "mode": "demo"
            })

        df = pd.DataFrame([data])

        # Aligner les colonnes
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_names]

        # Pipeline
        df_imputed = pd.DataFrame(imputer.transform(df),        columns=feature_names)
        df_scaled  = pd.DataFrame(scaler.transform(df_imputed), columns=feature_names)

        prediction = int(modele.predict(df_scaled)[0])
        proba      = round(float(modele.predict_proba(df_scaled)[0][1]), 3)

        return jsonify({
            "churn": prediction,
            "probabilite_churn": proba,
            "interpretation": "Client à risque" if prediction == 1 else "Client fidèle"
        })

    except Exception as e:
        return jsonify({"erreur": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)