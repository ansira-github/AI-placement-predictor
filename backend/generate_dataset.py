"""
Dataset Generator and Model Trainer for Placement Prediction
Generates 10,000 synthetic training samples and trains the model
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import traceback


def generate_placement_dataset(n_samples=10000, save_path='placement-dataset.csv'):
    """
    Generate synthetic placement data based on realistic patterns.
    
    Args:
        n_samples: Number of samples to generate
        save_path: Path to save the CSV file
    
    Returns:
        DataFrame with generated data
    """
    print(f"Generating {n_samples} samples...")
    np.random.seed(42)
    
    data = []
    
    for i in range(n_samples):
        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{n_samples} samples...")
        
        # Generate CGPA: Normal distribution centered at 6.5
        cgpa = np.random.normal(loc=6.5, scale=1.3)
        cgpa = np.clip(cgpa, 3.0, 10.0)
        
        # Generate IQ: Normal distribution centered at 110
        iq = np.random.normal(loc=110, scale=20)
        iq = np.clip(iq, 70, 160)
        
        # Placement decision logic
        cgpa_normalized = (cgpa - 3) / 7
        iq_normalized = (iq - 70) / 90
        
        placement_score = (
            cgpa_normalized * 0.6 + 
            iq_normalized * 0.4 + 
            np.random.normal(0, 0.15)
        )
        
        placed = 1 if placement_score > 0.5 else 0
        
        data.append({
            'cgpa': round(cgpa, 1),
            'iq': round(iq, 1),
            'placement': placed
        })
    
    df = pd.DataFrame(data)
    
    # Print statistics
    print(f"\n{'='*60}")
    print("DATASET STATISTICS")
    print(f"{'='*60}")
    print(f"Total samples: {len(df)}")
    print(f"Placement rate: {df['placement'].mean():.1%}")
    print(f"\nCGPA Statistics:")
    print(f"  Mean: {df['cgpa'].mean():.2f}")
    print(f"  Std: {df['cgpa'].std():.2f}")
    print(f"  Range: {df['cgpa'].min():.1f} - {df['cgpa'].max():.1f}")
    print(f"\nIQ Statistics:")
    print(f"  Mean: {df['iq'].mean():.2f}")
    print(f"  Std: {df['iq'].std():.2f}")
    print(f"  Range: {df['iq'].min():.1f} - {df['iq'].max():.1f}")
    
    placed_df = df[df['placement'] == 1]
    not_placed_df = df[df['placement'] == 0]
    
    print(f"\nPlacement Breakdown:")
    print(f"  Placed: {len(placed_df)} students")
    print(f"  Not Placed: {len(not_placed_df)} students")
    print(f"\nAverage Metrics:")
    print(f"  Placed - CGPA: {placed_df['cgpa'].mean():.2f}, IQ: {placed_df['iq'].mean():.2f}")
    print(f"  Not Placed - CGPA: {not_placed_df['cgpa'].mean():.2f}, IQ: {not_placed_df['iq'].mean():.2f}")
    print(f"{'='*60}\n")
    
    # Save to CSV
    df.to_csv(save_path, index=False)
    print(f"Dataset saved to: {save_path}")
    
    return df


def train_initial_model(dataset_path='placement-dataset.csv', model_path='placement_model.pkl'):
    """
    Train the placement prediction model.
    
    Args:
        dataset_path: Path to the dataset CSV
        model_path: Path to save the trained model
    """
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import classification_report, accuracy_score
        import joblib
    except ImportError as e:
        print(f"\nERROR: Missing required library!")
        print(f"Please install: pip install scikit-learn")
        print(f"Error details: {e}")
        sys.exit(1)
    
    print("\nLoading dataset...")
    df = pd.read_csv(dataset_path)
    
    X = df[['cgpa', 'iq']]
    y = df['placement']
    
    print("Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*60}")
    print("MODEL PERFORMANCE")
    print(f"{'='*60}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Placed', 'Placed']))
    print(f"{'='*60}\n")
    
    # Save model
    model_data = {
        'model': model,
        'scaler': scaler,
        'accuracy': accuracy,
        'train_samples': len(X_train)
    }
    
    import joblib
    joblib.dump(model_data, model_path)
    print(f"Model saved to: {model_path}\n")
    
    return model, scaler, accuracy


def main():
    """Main execution function."""
    try:
        print("\n" + "="*60)
        print("PLACEMENT PREDICTION SYSTEM - SETUP")
        print("="*60 + "\n")
        
        # Check if we're in the right directory
        current_dir = Path.cwd()
        print(f"Current directory: {current_dir}")
        
        # Determine paths based on current directory
        if current_dir.name == 'backend':
            # Running from backend folder
            dataset_path = 'placement-dataset.csv'
            model_path = 'placement_model.pkl'
            print("Running from backend folder")
        else:
            # Running from project root
            dataset_path = 'backend/placement-dataset.csv'
            model_path = 'backend/placement_model.pkl'
            print("Running from project root")
        
        print()
        
        # Step 1: Generate dataset
        print("STEP 1: Generating training data...")
        print("-" * 60)
        df = generate_placement_dataset(n_samples=10000, save_path=dataset_path)
        
        # Step 2: Train model
        print("\nSTEP 2: Training model...")
        print("-" * 60)
        model, scaler, accuracy = train_initial_model(dataset_path, model_path)
        
        # Success message
        print("\n" + "="*60)
        print("SETUP COMPLETE!")
        print("="*60)
        print(f"\nYour model has been trained on 10,000 samples")
        print(f"Final accuracy: {accuracy*100:.2f}%")
        print(f"\nFiles created:")
        print(f"  1. {dataset_path}")
        print(f"  2. {model_path}")
        print(f"\nYou can now run your Streamlit app:")
        print(f"  streamlit run frontend/app.py")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n{'='*60}")
        print("ERROR OCCURRED")
        print(f"{'='*60}")
        print(f"Error: {e}")
        print(f"\nFull traceback:")
        traceback.print_exc()
        print(f"{'='*60}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()