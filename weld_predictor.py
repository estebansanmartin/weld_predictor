"""
Weld Quality Predictor
Sistema predittivo qualità saldature per celle robotiche ABB
Industry 4.0 - Predictive Quality Control
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, 
                            roc_curve, auc, precision_recall_curve)
from dataclasses import dataclass
from typing import List, Dict, Tuple
import json
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("RdYlGn_r")

# ============ GENERATORE DATI REALISTICI ============

@dataclass
class WeldParameters:
    """Parametri di processo saldatura"""
    corrente: float          # A (100-300)
    tensione: float          # V (20-40)
    velocita: float          # mm/s (5-50)
    flusso_gas: float        # l/min (10-25)
    temperatura: float       # °C ambiente (15-35)
    spessore_materiale: float # mm (1-10)
    tipo_giunto: str         # 'lap', 'butt', 'fillet'
    posizione: str           # 'flat', 'vertical', 'overhead'

class WeldDataGenerator:
    """
    Genera dataset realistici di saldature con regole fisiche
    per qualità (buona/difettosa/porosa/cracked)
    """
    
    def __init__(self, n_samples: int = 2000):
        self.n_samples = n_samples
        
    def _calculate_quality_score(self, params: WeldParameters) -> float:
        """
        Calcola score qualità basato su regole fisiche saldatura MIG/MAG
        Ritorna valore 0-1 (1 = perfetta)
        """
        score = 1.0
        
        # 1. Rapporto corrente/spessore (regola: 30-40A per mm)
        ideal_current = params.spessore_materiale * 35
        current_dev = abs(params.corrente - ideal_current) / ideal_current
        score -= current_dev * 0.3
        
        # 2. Tensione vs corrente (arco stabile: V = 14 + 0.05*I)
        ideal_voltage = 14 + 0.05 * params.corrente
        volt_dev = abs(params.tensione - ideal_voltage) / ideal_voltage
        score -= volt_dev * 0.25
        
        # 3. Velocità di avanzamento
        wire_speed_factor = params.corrente / 150
        ideal_weld_speed = 20 + wire_speed_factor * 15
        speed_dev = abs(params.velocita - ideal_weld_speed) / ideal_weld_speed
        score -= speed_dev * 0.2
        
        # 4. Flusso gas protettivo
        if params.flusso_gas < 12:
            score -= 0.15
        elif params.flusso_gas > 22:
            score -= 0.1
        
        # 5. Penalità posizione difficile
        position_penalty = {
            'flat': 0,
            'vertical': 0.1,
            'overhead': 0.15
        }
        score -= position_penalty.get(params.posizione, 0)
        
        # 6. Temperatura ambiente
        if params.temperatura < 10 or params.temperatura > 30:
            score -= 0.1
        
        return max(0, min(1, score + np.random.normal(0, 0.05)))
    
    def generate(self) -> pd.DataFrame:
        """Genera dataset completo"""
        np.random.seed(42)
        records = []
        
        giunti = ['lap', 'butt', 'fillet']
        posizioni = ['flat', 'vertical', 'overhead']
        
        for _ in range(self.n_samples):
            spessore = np.random.uniform(1, 10)
            corrente = np.random.uniform(100, 300)
            
            params = WeldParameters(
                corrente=corrente,
                tensione=np.random.uniform(20, 40),
                velocita=np.random.uniform(5, 50),
                flusso_gas=np.random.uniform(10, 25),
                temperatura=np.random.uniform(15, 35),
                spessore_materiale=spessore,
                tipo_giunto=np.random.choice(giunti),
                posizione=np.random.choice(posizioni)
            )
            
            quality_score = self._calculate_quality_score(params)
            
            if quality_score > 0.85:
                qualita = 'eccellente'
                difetto = 'nessuno'
            elif quality_score > 0.70:
                qualita = 'buona'
                difetto = 'nessuno'
            elif quality_score > 0.55:
                qualita = 'accettabile'
                difetto = np.random.choice(['porosità_minima', 'rinforzo_eccessivo'])
            elif quality_score > 0.40:
                qualita = 'scarsa'
                difetto = np.random.choice(['porosità', 'inclusioni', 'mancata_fusione'])
            else:
                qualita = 'difettosa'
                difetto = np.random.choice(['crack', 'porosità_grave', 'penetrazione_insufficiente'])
            
            record = {
                'corrente_A': params.corrente,
                'tensione_V': params.tensione,
                'velocita_mm_s': params.velocita,
                'flusso_gas_l_min': params.flusso_gas,
                'temperatura_C': params.temperatura,
                'spessore_mm': params.spessore_materiale,
                'tipo_giunto': params.tipo_giunto,
                'posizione': params.posizione,
                'quality_score': round(quality_score, 3),
                'qualita': qualita,
                'difetto': difetto,
                'target_ok': 1 if quality_score > 0.70 else 0
            }
            records.append(record)
        
        return pd.DataFrame(records)

# ============ ANALISI ESPLORATIVA ============

class WeldQualityAnalyzer:
    """Analisi esplorativa e feature engineering"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.feature_cols = ['corrente_A', 'tensione_V', 'velocita_mm_s', 
                           'flusso_gas_l_min', 'temperatura_C', 'spessore_mm']
        
    def correlation_analysis(self) -> pd.DataFrame:
        """Correlazione parametri vs qualità"""
        df_encoded = self.df.copy()
        df_encoded['tipo_giunto_encoded'] = LabelEncoder().fit_transform(df_encoded['tipo_giunto'])
        df_encoded['posizione_encoded'] = LabelEncoder().fit_transform(df_encoded['posizione'])
        
        corr_data = df_encoded[self.feature_cols + ['tipo_giunto_encoded', 
                                                   'posizione_encoded', 'quality_score']]
        return corr_data.corr()['quality_score'].sort_values(ascending=False)
    
    def feature_importance_viz(self, save_path: str = "feature_importance.png"):
        """Visualizza importanza parametri"""
        corr = self.correlation_analysis().drop('quality_score').abs().sort_values()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.RdYlGn(corr.values)
        bars = ax.barh(corr.index, corr.values, color=colors)
        ax.set_xlabel('Importanza (|Correlazione|)')
        ax.set_title('Feature Importance - Parametri Critici Qualità', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        
        for i, (bar, val) in enumerate(zip(bars, corr.values)):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{val:.2f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Salvato: {save_path}")
    
    def process_window_analysis(self, save_path: str = "process_windows.png"):
        """Finestre operative ottimali"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, col in enumerate(self.feature_cols):
            ax = axes[idx]
            
            data_to_plot = [self.df[self.df['qualita'] == q][col].values 
                          for q in ['eccellente', 'buona', 'accettabile', 'scarsa', 'difettosa']]
            
            bp = ax.boxplot(data_to_plot, labels=['Ecc', 'Buo', 'Acc', 'Sca', 'Dif'], patch_artist=True)
            
            colors = ['#2ecc71', '#27ae60', '#f1c40f', '#e67e22', '#e74c3c']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_title(f'{col}', fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Process Windows by Quality Level', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Salvato: {save_path}")

# ============ MACHINE LEARNING ============

class WeldQualityPredictor:
    """Modello predittivo qualità saldature"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.models = {}
        self.scaler = StandardScaler()
        self.le_giunto = LabelEncoder()
        self.le_posizione = LabelEncoder()
        
    def _prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Feature engineering e encoding"""
        df = df.copy()
        
        df['tipo_giunto_enc'] = self.le_giunto.fit_transform(df['tipo_giunto'])
        df['posizione_enc'] = self.le_posizione.fit_transform(df['posizione'])
        
        df['potenza_W'] = df['corrente_A'] * df['tensione_V']
        df['energia_specifica'] = df['potenza_W'] / (df['velocita_mm_s'] * df['spessore_mm'])
        df['rapporto_corrente_spessore'] = df['corrente_A'] / df['spessore_mm']
        
        feature_cols = ['corrente_A', 'tensione_V', 'velocita_mm_s', 'flusso_gas_l_min',
                       'temperatura_C', 'spessore_mm', 'tipo_giunto_enc', 'posizione_enc',
                       'potenza_W', 'energia_specifica', 'rapporto_corrente_spessore']
        
        return df[feature_cols].values, feature_cols
    
    def train(self):
        """Allena e confronta modelli"""
        X, feature_names = self._prepare_features(self.df)
        y = self.df['target_ok'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        candidates = {
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
        }
        
        print("\nTraining modelli...")
        print("=" * 50)
        
        best_score = 0
        for name, model in candidates.items():
            if name == 'Logistic Regression':
                scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
                X_test_final = X_test_scaled
            else:
                scores = cross_val_score(model, X_train, y_train, cv=5)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
                X_test_final = X_test
            
            accuracy = scores.mean()
            self.models[name] = {
                'model': model,
                'accuracy': accuracy,
                'X_test': X_test_final,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_prob': y_prob,
                'feature_names': feature_names
            }
            
            print(f"{name:20} | CV Accuracy: {accuracy:.3f}")
            
            if accuracy > best_score:
                best_score = accuracy
                self.best_model_name = name
        
        print(f"\nBest Model: {self.best_model_name} ({best_score:.3f})")
        return self.models
    
    def evaluate(self, save_path: str = "model_evaluation.png"):
        """Valutazione completa best model"""
        from datetime import datetime
        
        model_data = self.models[self.best_model_name]
        y_test = model_data['y_test']
        y_pred = model_data['y_pred']
        y_prob = model_data['y_prob']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
        axes[0,0].set_title(f'Confusion Matrix - {self.best_model_name}', fontweight='bold')
        axes[0,0].set_xlabel('Predetto')
        axes[0,0].set_ylabel('Reale')
        
        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        axes[0,1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        axes[0,1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0,1].set_xlim([0.0, 1.0])
        axes[0,1].set_ylim([0.0, 1.05])
        axes[0,1].set_xlabel('False Positive Rate')
        axes[0,1].set_ylabel('True Positive Rate')
        axes[0,1].set_title('ROC Curve', fontweight='bold')
        axes[0,1].legend(loc="lower right")
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Precision-Recall
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        axes[1,0].plot(recall, precision, color='blue', lw=2)
        axes[1,0].set_xlabel('Recall')
        axes[1,0].set_ylabel('Precision')
        axes[1,0].set_title('Precision-Recall Curve', fontweight='bold')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Feature Importance
        model = model_data['model']
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            importances = np.abs(model.coef_[0])
        
        indices = np.argsort(importances)[::-1][:8]
        axes[1,1].bar(range(len(indices)), importances[indices], color='green', alpha=0.7)
        axes[1,1].set_xticks(range(len(indices)))
        axes[1,1].set_xticklabels([model_data['feature_names'][i] for i in indices], 
                                 rotation=45, ha='right', fontsize=9)
        axes[1,1].set_title('Top Feature Importance', fontweight='bold')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Salvato: {save_path}")
        
        print(f"\nClassification Report - {self.best_model_name}:")
        print(classification_report(y_test, y_pred, target_names=['Difettosa', 'Buona']))
        
        return roc_auc
    
    def predict_single(self, params: WeldParameters) -> Dict:
        """Predice qualità per singola saldatura"""
        input_data = pd.DataFrame([{
            'corrente_A': params.corrente,
            'tensione_V': params.tensione,
            'velocita_mm_s': params.velocita,
            'flusso_gas_l_min': params.flusso_gas,
            'temperatura_C': params.temperatura,
            'spessore_mm': params.spessore_materiale,
            'tipo_giunto': params.tipo_giunto,
            'posizione': params.posizione
        }])
        
        X, _ = self._prepare_features(input_data)
        
        model_data = self.models[self.best_model_name]
        model = model_data['model']
        
        if self.best_model_name == 'Logistic Regression':
            X = self.scaler.transform(X)
        
        prob = model.predict_proba(X)[0]
        prediction = model.predict(X)[0]
        
        suggestions = self._generate_suggestions(params, prob[1])
        
        return {
            'qualita_prevista': 'Buona' if prediction == 1 else 'Difettosa',
            'confidenza': f"{max(prob)*100:.1f}%",
            'probabilita_buona': f"{prob[1]*100:.1f}%",
            'suggerimenti': suggestions
        }
    
    def _generate_suggestions(self, params: WeldParameters, prob_good: float) -> List[str]:
        """Genera suggerimenti per migliorare qualità"""
        suggestions = []
        
        if prob_good < 0.7:
            ideal_current = params.spessore_materiale * 35
            if abs(params.corrente - ideal_current) > 20:
                suggestions.append(f"Corrente ottimale: {ideal_current:.0f}A (attuale: {params.corrente:.0f}A)")
            
            ideal_voltage = 14 + 0.05 * params.corrente
            if abs(params.tensione - ideal_voltage) > 2:
                suggestions.append(f"Tensione ottimale: {ideal_voltage:.1f}V (attuale: {params.tensione:.1f}V)")
            
            if params.flusso_gas < 12:
                suggestions.append("Aumentare flusso gas protettivo (>12 l/min)")
            elif params.flusso_gas > 22:
                suggestions.append("Ridurre flusso gas per evitare turbolenza")
            
            if params.velocita > 35:
                suggestions.append("Ridurre velocità per migliorare penetrazione")
            elif params.velocita < 10:
                suggestions.append("Aumentare velocità per evitare gocciolamento")
        
        return suggestions if suggestions else ["Parametri ottimali"]

# ============ DASHBOARD ============

class QualityDashboard:
    """Dashboard qualità saldature"""
    
    def __init__(self, df: pd.DataFrame, predictor: WeldQualityPredictor):
        self.df = df
        self.predictor = predictor
        
    def generate_quality_distribution(self, save_path: str = "quality_distribution.png"):
        """Distribuzione qualità dataset"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        qual_counts = self.df['qualita'].value_counts()
        colors = {'eccellente': '#2ecc71', 'buona': '#27ae60', 'accettabile': '#f1c40f',
                 'scarsa': '#e67e22', 'difettosa': '#e74c3c'}
        
        axes[0].pie(qual_counts.values, labels=qual_counts.index, autopct='%1.1f%%',
                   colors=[colors.get(q, 'gray') for q in qual_counts.index])
        axes[0].set_title('Distribuzione Qualità Saldature', fontweight='bold')
        
        difetti = self.df[self.df['difetto'] != 'nessuno']['difetto'].value_counts().head(6)
        axes[1].barh(difetti.index, difetti.values, color='coral')
        axes[1].set_title('Difetti più Frequenti', fontweight='bold')
        axes[1].set_xlabel('Conteggio')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Salvato: {save_path}")
    
    def generate_optimization_map(self, save_path: str = "optimization_map.png"):
        """Mappa ottimizzazione parametri"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Corrente vs Tensione
        scatter = axes[0,0].scatter(self.df['corrente_A'], self.df['tensione_V'], 
                                   c=self.df['quality_score'], cmap='RdYlGn', 
                                   alpha=0.6, s=30)
        plt.colorbar(scatter, ax=axes[0,0], label='Quality Score')
        axes[0,0].set_xlabel('Corrente (A)')
        axes[0,0].set_ylabel('Tensione (V)')
        axes[0,0].set_title('Finestra Operativa Corrente-Tensione', fontweight='bold')
        
        # Velocità vs Spessore
        for qual in ['eccellente', 'buona', 'difettosa']:
            subset = self.df[self.df['qualita'] == qual]
            axes[0,1].scatter(subset['spessore_mm'], subset['velocita_mm_s'], 
                            label=qual, alpha=0.6, s=30)
        axes[0,1].set_xlabel('Spessore (mm)')
        axes[0,1].set_ylabel('Velocità (mm/s)')
        axes[0,1].set_title('Velocità vs Spessore per Qualità', fontweight='bold')
        axes[0,1].legend()
        
        # Temperatura ambiente
        axes[1,0].boxplot([self.df[self.df['qualita'] == q]['temperatura_C'].values 
                         for q in ['eccellente', 'buona', 'difettosa']],
                        labels=['Eccellente', 'Buona', 'Difettosa'])
        axes[1,0].set_ylabel('Temperatura Ambiente (°C)')
        axes[1,0].set_title('Effetto Temperatura su Qualità', fontweight='bold')
        
        # Flusso gas
        axes[1,1].hist(self.df[self.df['qualita'] == 'eccellente']['flusso_gas_l_min'], 
                      bins=20, alpha=0.7, label='Eccellente', color='green')
        axes[1,1].hist(self.df[self.df['qualita'] == 'difettosa']['flusso_gas_l_min'], 
                      bins=20, alpha=0.7, label='Difettosa', color='red')
        axes[1,1].set_xlabel('Flusso Gas (l/min)')
        axes[1,1].set_ylabel('Frequenza')
        axes[1,1].set_title('Distribuzione Flusso Gas', fontweight='bold')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Salvato: {save_path}")

# ============ MAIN ============

def main():
    print("Weld Quality Predictor")
    print("Predictive Quality Control for ABB Welding Cells")
    print("=" * 60)
    
    # 1. Genera dati
    print("\nGenerazione dataset storico saldature...")
    generator = WeldDataGenerator(n_samples=3000)
    df = generator.generate()
    print(f"Generati {len(df)} record di saldature")
    print(f"Distribuzione qualità:")
    for qual, count in df['qualita'].value_counts().items():
        print(f"  {qual:12}: {count:4d} ({count/len(df)*100:.1f}%)")
    
    # 2. Analisi esplorativa
    print("\nAnalisi parametri critici...")
    analyzer = WeldQualityAnalyzer(df)
    corr = analyzer.correlation_analysis()
    print("Top 3 parametri correlati alla qualità:")
    for param, val in corr.head(3).items():
        print(f"  {param:20}: {val:+.3f}")
    
    analyzer.feature_importance_viz("outputs/feature_importance.png")
    analyzer.process_window_analysis("outputs/process_windows.png")
    
    # 3. Machine Learning
    print("\nTraining modelli predittivi...")
    predictor = WeldQualityPredictor(df)
    models = predictor.train()
    auc_score = predictor.evaluate("outputs/model_evaluation.png")
    
    # 4. Dashboard
    print("\nGenerazione dashboard...")
    dashboard = QualityDashboard(df, predictor)
    dashboard.generate_quality_distribution("outputs/quality_distribution.png")
    dashboard.generate_optimization_map("outputs/optimization_map.png")
    
    # 5. Demo predizione
    print("\nDemo predizione singola saldatura:")
    test_params = WeldParameters(
        corrente=180, tensione=23.5, velocita=25,
        flusso_gas=15, temperatura=22, spessore_materiale=5,
        tipo_giunto='butt', posizione='vertical'
    )
    result = predictor.predict_single(test_params)
    print(f"Input: {test_params.corrente}A, {test_params.tensione}V, {test_params.velocita}mm/s")
    print(f"Predizione: {result['qualita_prevista']} (confidenza: {result['confidenza']})")
    print(f"Suggerimenti:")
    for sug in result['suggerimenti']:
        print(f"  {sug}")
    
    # 6. Salva
    df.to_csv("outputs/weld_data.csv", index=False)
    
    summary = {
        'dataset_size': len(df),
        'good_quality_pct': (df['target_ok'].sum() / len(df) * 100),
        'best_model': predictor.best_model_name,
        'model_accuracy': models[predictor.best_model_name]['accuracy'],
        'auc_score': auc_score,
        'top_feature': corr.index[0],
        'generated_at': datetime.now().isoformat()
    }
    
    with open("outputs/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("ANALISI COMPLETATA")
    print("=" * 60)
    print(f"\nOutput in /outputs:")
    print("  • feature_importance.png - Parametri critici")
    print("  • process_windows.png - Finestre operative")
    print("  • model_evaluation.png - Performance ML")
    print("  • quality_distribution.png - Distribuzione qualità")
    print("  • optimization_map.png - Mappa ottimizzazione")
    print("  • weld_data.csv - Dataset completo")
    print("  • summary.json - Metriche riassuntive")
    print(f"\nAccuratezza modello: {summary['model_accuracy']:.1%}")
    print(f"AUC Score: {auc_score:.3f}")

if __name__ == "__main__":
    from datetime import datetime
    import os
    os.makedirs('outputs', exist_ok=True)
    main()