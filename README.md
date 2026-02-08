# Weld Quality Predictor

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


## Requisiti

```bash
pip install pandas numpy matplotlib seaborn scikit-learn

