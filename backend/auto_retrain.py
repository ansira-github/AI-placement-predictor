import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import json


class AutoRetrainer:
    """
    Automatically retrains the placement prediction model with new data.
    """
    
    def __init__(self, base_dataset_path, model_save_path, config_path='training_config.json'):
        self.base_dataset_path = base_dataset_path
        self.model_save_path = model_save_path
        self.config_path = config_path
        self.config = self.load_config()
        
    def load_config(self):
        """Load or create training configuration."""
        default_config = {
            'min_samples_for_retrain': 50,  # Minimum new samples before retraining
            'test_size': 0.2,
            'min_accuracy_threshold': 0.75,  # Don't deploy if accuracy drops below this
            'backup_old_model': True,
            'last_train_date': None,
            'total_samples_trained': 0,
            'model_version': 1
        }
        
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                # Merge with defaults for any missing keys
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
        else:
            return default_config
    
    def save_config(self):
        """Save training configuration."""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def load_base_dataset(self):
        """Load the base training dataset."""
        if not os.path.exists(self.base_dataset_path):
            raise FileNotFoundError(f"Base dataset not found: {self.base_dataset_path}")
        return pd.read_csv(self.base_dataset_path)
    
    def validate_new_data(self, new_df):
        """Validate new data before adding to training set."""
        required_cols = {'cgpa', 'iq', 'placement'}
        
        if not required_cols.issubset(new_df.columns):
            raise ValueError(f"New data must contain columns: {required_cols}")
        
        # Check for valid ranges
        if not ((new_df['cgpa'] >= 0) & (new_df['cgpa'] <= 10)).all():
            raise ValueError("CGPA values must be between 0 and 10")
        
        if not ((new_df['iq'] >= 50) & (new_df['iq'] <= 200)).all():
            raise ValueError("IQ values must be between 50 and 200")
        
        if not new_df['placement'].isin([0, 1]).all():
            raise ValueError("Placement values must be 0 or 1")
        
        return True
    
    def append_new_data(self, new_data_path):
        """
        Append new verified placement data to the base dataset.
        
        Args:
            new_data_path: Path to CSV with new placement records
            
        Returns:
            int: Number of new samples added
        """
        base_df = self.load_base_dataset()
        new_df = pd.read_csv(new_data_path)
        
        # Validate
        self.validate_new_data(new_df)
        
        # Remove duplicates
        initial_count = len(new_df)
        combined_df = pd.concat([base_df, new_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['cgpa', 'iq'], keep='last')
        
        new_samples_added = len(combined_df) - len(base_df)
        
        # Save updated dataset
        combined_df.to_csv(self.base_dataset_path, index=False)
        
        print(f"Added {new_samples_added} new samples (filtered {initial_count - new_samples_added} duplicates)")
        print(f"Total dataset size: {len(combined_df)}")
        
        return new_samples_added
    
    def train_models(self, df):
        """Train multiple models and select the best."""
        X = df[['cgpa', 'iq']]
        y = df['placement']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config['test_size'], random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        models = {
            'logistic': LogisticRegression(max_iter=1000, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            'gradient_boost': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'auc': auc,
                'scaler': scaler
            }
        
        # Select best model
        best_name = max(results.keys(), key=lambda k: results[k]['auc'])
        best_result = results[best_name]
        
        return best_result, best_name, results
    
    def backup_model(self):
        """Create backup of current model."""
        if os.path.exists(self.model_save_path):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = self.model_save_path.replace('.pkl', f'_backup_{timestamp}.pkl')
            os.rename(self.model_save_path, backup_path)
            print(f"Backed up old model to: {backup_path}")
    
    def retrain(self, new_data_path=None, force=False):
        """
        Main retraining function.
        
        Args:
            new_data_path: Path to new data CSV (optional)
            force: Force retrain even if below minimum sample threshold
            
        Returns:
            dict: Training results and metrics
        """
        print("\n" + "="*70)
        print("AUTO-RETRAINING SYSTEM")
        print("="*70)
        
        # Add new data if provided
        new_samples = 0
        if new_data_path:
            new_samples = self.append_new_data(new_data_path)
        
        # Check if retraining is needed
        if not force and new_samples < self.config['min_samples_for_retrain']:
            print(f"\nInsufficient new samples ({new_samples}/{self.config['min_samples_for_retrain']})")
            print("Skipping retraining. Use force=True to retrain anyway.")
            return {'status': 'skipped', 'reason': 'insufficient_samples'}
        
        # Load full dataset
        df = self.load_base_dataset()
        print(f"\nTraining on {len(df)} total samples...")
        
        # Train models
        best_result, best_name, all_results = self.train_models(df)
        
        # Display results
        print(f"\nModel Performance:")
        print("-" * 70)
        for name, result in all_results.items():
            marker = " <- BEST" if name == best_name else ""
            print(f"{name:20} | Accuracy: {result['accuracy']:.4f} | AUC: {result['auc']:.4f}{marker}")
        
        # Check accuracy threshold
        if best_result['accuracy'] < self.config['min_accuracy_threshold']:
            print(f"\nWARNING: Best model accuracy ({best_result['accuracy']:.4f}) below threshold")
            print(f"Threshold: {self.config['min_accuracy_threshold']:.4f}")
            print("Model NOT saved. Check your data quality.")
            return {'status': 'failed', 'reason': 'accuracy_below_threshold', 'accuracy': best_result['accuracy']}
        
        # Backup old model
        if self.config['backup_old_model']:
            self.backup_model()
        
        # Save new model
        model_data = {
            'model': best_result['model'],
            'scaler': best_result['scaler'],
            'model_name': best_name,
            'accuracy': best_result['accuracy'],
            'auc': best_result['auc'],
            'train_date': datetime.now().isoformat(),
            'total_samples': len(df),
            'version': self.config['model_version'] + 1
        }
        
        joblib.dump(model_data, self.model_save_path)
        
        # Update config
        self.config['last_train_date'] = datetime.now().isoformat()
        self.config['total_samples_trained'] = len(df)
        self.config['model_version'] += 1
        self.save_config()
        
        print(f"\nModel saved: {self.model_save_path}")
        print(f"Version: {self.config['model_version']}")
        print(f"Accuracy: {best_result['accuracy']:.4f}")
        print(f"AUC: {best_result['auc']:.4f}")
        print("="*70)
        
        return {
            'status': 'success',
            'model_name': best_name,
            'accuracy': best_result['accuracy'],
            'auc': best_result['auc'],
            'version': self.config['model_version'],
            'total_samples': len(df)
        }
    
    def schedule_retrain_check(self):
        """Check if scheduled retraining is needed based on config."""
        if self.config['last_train_date'] is None:
            return True
        
        last_train = datetime.fromisoformat(self.config['last_train_date'])
        days_since_train = (datetime.now() - last_train).days
        
        # Retrain if more than 30 days since last training
        return days_since_train >= 30


# ============================================
# USAGE EXAMPLES
# ============================================

def example_usage():
    """Example of how to use the auto-retrainer."""
    
    # Initialize retrainer
    retrainer = AutoRetrainer(
        base_dataset_path='backend/placement-dataset.csv',
        model_save_path='backend/placement_model_advanced.pkl'
    )
    
    # Example 1: Retrain with new data
    print("\n--- Example 1: Adding New Data ---")
    result = retrainer.retrain(new_data_path='new_placement_data.csv')
    print(f"Result: {result}")
    
    # Example 2: Force retrain without new data
    print("\n--- Example 2: Force Retrain ---")
    result = retrainer.retrain(force=True)
    print(f"Result: {result}")
    
    # Example 3: Check if scheduled retrain is needed
    print("\n--- Example 3: Scheduled Check ---")
    if retrainer.schedule_retrain_check():
        print("Scheduled retraining needed!")
        result = retrainer.retrain(force=True)
    else:
        print("No retraining needed yet.")


def create_sample_new_data():
    """Create a sample new data CSV for testing."""
    new_data = {
        'cgpa': [7.8, 6.5, 8.2, 5.9, 7.1],
        'iq': [125, 105, 135, 95, 115],
        'placement': [1, 0, 1, 0, 1]
    }
    df = pd.DataFrame(new_data)
    df.to_csv('new_placement_data_sample.csv', index=False)
    print("Sample new data created: new_placement_data_sample.csv")


if __name__ == "__main__":
    print("\nAUTO-RETRAINER SYSTEM - DEMO")
    print("="*70)
    
    # Check if base dataset exists
    base_path = 'backend/placement-dataset.csv'
    if not os.path.exists(base_path):
        print(f"\nERROR: Base dataset not found at {base_path}")
        print("Please run the dataset generator first!")
    else:
        # Initialize
        retrainer = AutoRetrainer(
            base_dataset_path=base_path,
            model_save_path='backend/placement_model_advanced.pkl'
        )
        
        # Show current config
        print("\nCurrent Configuration:")
        for key, value in retrainer.config.items():
            print(f"  {key}: {value}")
        
        # Create sample new data for testing
        print("\nCreating sample new data for testing...")
        create_sample_new_data()
        
        # Run retrain with sample data
        print("\nRunning retraining with sample data...")
        result = retrainer.retrain(new_data_path='new_placement_data_sample.csv', force=True)
        
        print("\nRetraining Result:")
        for key, value in result.items():
            print(f"  {key}: {value}")
        
        print("\n" + "="*70)
        print("Demo complete! Check training_config.json for configuration.")