# 🔥 Weld Quality Predictor

Sistema di Predictive Quality Control per saldature robotiche ABB basato su Machine Learning.

## Panoramica

Piattaforma che predice la qualità di una saldatura **prima** che venga eseguita, analizzando i parametri di processo (corrente, tensione, velocità, gas, ecc.). Riduce scarti e ottimizza parametri in tempo reale.

## Problema Industriale

Nelle celle di saldatura tradizionali:
- ❌ Il controllo qualità avviene **dopo** la saldatura (ispezione visiva/raggi X)
- ❌ Gli scarti si accumulano prima di intervenire sui parametri
- ❌ L'ottimizzazione è trial-and-error basata su esperienza operatore

**Soluzione**: Modello ML che predice la probabilità di difetto con **85%+ accuratezza**, suggerendo parametri ottimali in anticipo.

## Architettura

Parametri Processo → Feature Engineering → ML Classifier → Predizione Qualità
(A, V, mm/s)      (Potenza, Energia,      (Random Forest)    + Suggerimenti
Rapporti fisici)                           Ottimizzazione


## Dataset

| Feature | Unità | Range | Descrizione |
|---------|-------|-------|-------------|
| `corrente_A` | A | 100-300 | Corrente saldatura |
| `tensione_V` | V | 20-40 | Tensione arco |
| `velocita_mm_s` | mm/s | 5-50 | Velocità avanzamento |
| `flusso_gas_l_min` | l/min | 10-25 | Flusso gas protettivo |
| `temperatura_C` | °C | 15-35 | Temperatura ambiente |
| `spessore_mm` | mm | 1-10 | Spessore materiale base |
| `tipo_giunto` | - | lap/butt/fillet | Geometria giunto |
| `posizione` | - | flat/vertical/overhead | Posizione saldatura |

**Target**: `qualita` (eccellente/buona/accettabile/scarsa/difettosa) → binario `target_ok`

## Modelli Testati

| Modello | Accuracy | Caratteristiche |
|---------|----------|-----------------|
| **Random Forest** | ~87% | Best overall, gestisce non-linearità |
| Gradient Boosting | ~85% | Buono per feature importance |
| Logistic Regression | ~82% | Interpretabile, baseline |

## Feature Engineering

Oltre ai parametri grezzi, il modello utilizza:
- `potenza_W` = Corrente × Tensione
- `energia_specifica` = Potenza / (Velocità × Spessore)
- `rapporto_corrente_spessore` = Corrente / Spessore (regola 35A/mm)

## Utilizzo

Training e valutazione
```
python weld_predictor.py
```
Predizione singola (integrazione)
```
from weld_predictor import WeldQualityPredictor, WeldParameters

# Carica modello addestrato
predictor = WeldQualityPredictor(df)
predictor.train()

# Predici nuova saldatura
params = WeldParameters(
    corrente=180,
    tensione=23.5,
    velocita=25,
    flusso_gas=15,
    temperatura=22,
    spessore_materiale=5,
    tipo_giunto='butt',
    posizione='vertical'
)

result = predictor.predict_single(params)
# Output: {'qualita_prevista': 'Buona', 'confidenza': '89.3%', ...}
```

## Insight Chiave

# Parametri più influenti (dal modello)

Rapporto corrente/spessore (0.42) - Regola dei 35A/mm
Energia specifica (0.38) - Bilanciamento calore/input
Tensione (0.31) - Stabilità arco
Velocità (0.28) - Penetrazione vs gocciolamento
Regole estratte
Corrente ottimale: 30-40A per mm di spessore
Tensione arco stabile: V = 14 + 0.05 × I
Velocità: 20-35 mm/s per spessori 3-6mm
Flusso gas: 12-20 l/min (evitare <12 porosità, >22 turbolenza)

## Integrazione Industriale con controller ABB
```
# Pseudocodice integrazione RAPID ↔ Python
def pre_weld_check(corrente, tensione, velocita, ...):
    prediction = ml_model.predict([corrente, tensione, ...])
    if prediction['probabilita_buona'] < 0.7:
        # Alert operatore o auto-correzione
        adjust_parameters(prediction['suggerimenti'])
```

## Requisiti

```
pip install pandas numpy matplotlib seaborn scikit-learn

