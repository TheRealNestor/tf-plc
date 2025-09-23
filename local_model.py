import tensorflow as tf
import numpy as np
import os
import pandas as pd
from datetime import datetime
from pathlib import Path

class TemperaturePredictionModel:
    def __init__(self, sequence_length=5, model_name=None):  # Small window for real-time
        self.model = None
        self.sequence_length = sequence_length

        # Create models directory if it doesn't exist
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"temp_sensor_{timestamp}"
        
        self.model_path = self.models_dir / f"{model_name}.keras"
        
    def create_model(self):
        """Create and compile simple feedforward model for real-time temperature classification"""
        self.model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(self.sequence_length, 1)),  # Flatten the window
            tf.keras.layers.Dense(10, activation='relu'),  # First hidden layer with 10 neurons
            tf.keras.layers.Dense(10, activation='relu'),  # Second hidden layer with 10 neurons
            tf.keras.layers.Dense(3, activation='softmax')  # 3 classes: cold, normal, hot
        ])
        self.model.compile(
            optimizer='adam', 
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return self.model
    
    def generate_temperature_data(self, samples=1000):
        """Generate realistic temperature sensor data for demonstration"""
        np.random.seed(42)
        temperatures = []
        
        # Simulate different scenarios: room temp, cold water, hot water
        for i in range(samples):
            if i < samples // 3:
                # Room temperature (normal)
                temp = np.random.normal(20, 2)  # Around 20°C ± 2°C
            elif i < 2 * samples // 3:
                # Cold water scenario  
                temp = np.random.normal(5, 1)   # Around 5°C ± 1°C
            else:
                # Hot water scenario
                temp = np.random.normal(35, 2)  # Around 35°C ± 2°C
            
            temperatures.append(temp)
        
        return np.array(temperatures)
    
    def create_sequences(self, data):
        """Create training sequences for classification (fallback method)"""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            temp = data[i + self.sequence_length]
            # Convert temperature to class: 0=cold (<10°C), 1=normal (10-25°C), 2=hot (>25°C)
            if temp < 10:
                class_label = 0  # cold
            elif temp <= 25:
                class_label = 1  # normal
            else:
                class_label = 2  # hot
            
            # One-hot encode the class
            one_hot = [0, 0, 0]
            one_hot[class_label] = 1
            y.append(one_hot)
        
        return np.array(X).reshape(-1, self.sequence_length, 1), np.array(y)
    
    def create_sequences_from_labels(self, temperatures, labels):
        """Create training sequences from temperature data and labels"""
        X, y = [], []
        
        # Convert string labels to numeric
        label_map = {'cold': 0, 'normal': 1, 'hot': 2}
        
        for i in range(len(temperatures) - self.sequence_length):
            # Use sequence of temperatures as input
            X.append(temperatures[i:(i + self.sequence_length)])
            
            # Use the label of the target temperature as output
            target_label = labels[i + self.sequence_length]
            class_label = label_map.get(target_label, 1)  # Default to normal if unknown
            
            # One-hot encode the class
            one_hot = [0, 0, 0]
            one_hot[class_label] = 1
            y.append(one_hot)
        
        return np.array(X).reshape(-1, self.sequence_length, 1), np.array(y)
    
    def train_model(self, csv_file_path="models/temperature_data.csv", samples=1000):
        """Train the model using CSV data or generated data as fallback"""
        if self.model is None:
            self.create_model()
        
        # Try to load data from CSV first
        temperatures, labels = self.load_data_from_csv(csv_file_path)
        
        if temperatures is not None and labels is not None:
            print("Using CSV data for training...")
            X, y = self.create_sequences_from_labels(temperatures, labels)
        else:
            print("Using generated data for training...")
            temperatures = self.generate_temperature_data(samples)
            X, y = self.create_sequences(temperatures)
        
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"Training model with {len(X_train)} samples...")
        print(f"Input shape: {X_train.shape}")
        print(f"Output shape: {y_train.shape}")
        
        self.model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                      epochs=100, batch_size=16, verbose=1)
        
        print("Training complete!")
        return temperatures
    
    def predict(self, recent_temperatures, steps_ahead=1):
        """Predict temperature category from recent sensor readings"""
        if self.model is None or len(recent_temperatures) < self.sequence_length:
            return None
        
        sequence = np.array(recent_temperatures[-self.sequence_length:]).reshape(1, self.sequence_length, 1)
        predictions = []
        class_names = ['Cold (<10°C)', 'Normal (10-25°C)', 'Hot (>25°C)']
        
        for i in range(steps_ahead):
            pred_probs = self.model.predict(sequence, verbose=0)[0]
            predicted_class = np.argmax(pred_probs)
            confidence = pred_probs[predicted_class]
            
            prediction = {
                'class': predicted_class,
                'class_name': class_names[predicted_class],
                'confidence': confidence,
                'probabilities': {
                    'cold': pred_probs[0],
                    'normal': pred_probs[1], 
                    'hot': pred_probs[2]
                }
            }
            predictions.append(prediction)
            
            # For multi-step, shift the window (not very meaningful for sensor demo)
            if steps_ahead > 1:
                if predicted_class == 0:  # cold
                    next_temp = 5.0
                elif predicted_class == 1:  # normal
                    next_temp = 20.0
                else:  # hot
                    next_temp = 35.0
                    
                sequence = np.append(sequence[0][1:], next_temp).reshape(1, self.sequence_length, 1)
        
        return predictions
    
    def save_model(self):
        """Save the complete model in .keras format"""
        if not self.model:
            print("No model to save.")
            return False
        
        try:
            self.model.save(self.model_path)
            size_mb = self.model_path.stat().st_size / (1024 * 1024)
            print(f"Model saved to {self.model_path.name} ({size_mb:.2f} MB)")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def save_weights(self):
        """Save only the model weights using TensorFlow format"""
        if not self.model:
            print("No model to save weights from.")
            return False
        
        weights_path = Path(str(self.model_path).replace('.keras', '.weights.h5'))
        try:
            self.model.save_weights(weights_path, overwrite=True)
            print(f"Weights saved to {weights_path.name} (TensorFlow format)")
            return True
        except Exception as e:
            print(f"Error saving weights: {e}")
            return False
    
    def load_model(self):
        """Load the complete model from .keras format"""
        if self.model_path.exists():
            try:
                self.model = tf.keras.models.load_model(self.model_path)
                size_mb = self.model_path.stat().st_size / (1024 * 1024)
                print(f"Model loaded from {self.model_path.name} ({size_mb:.2f} MB)")
                return True
            except Exception as e:
                print(f"Failed to load model: {e}")
                return False
        
        print(f"Model file not found: {self.model_path.name}")
        return False
    
    def load_weights(self):
        """Load only weights (requires model architecture to be created first)"""
        if self.model is None:
            print("Model architecture must be created before loading weights")
            return False
            
        weights_path = Path(str(self.model_path).replace('.keras', '.weights.h5'))
        
        try:
            self.model.load_weights(weights_path)
            print(f"Weights loaded from {weights_path.name}")
            return True
        except Exception as e:
            print(f"Failed to load weights: {e}")
            return False
    
    def load_data_from_csv(self, csv_file_path="temperature_data.csv"):
        """Load temperature data from CSV file"""
        csv_path = Path(csv_file_path)
        try:
            # Try pandas first, fall back to manual parsing if not available
            try:
                df = pd.read_csv(csv_path)
                temperatures = df['temperature'].values
                labels = df['label'].values
            except ImportError:
                # Manual CSV parsing if pandas not available
                temperatures = []
                labels = []
                with csv_path.open('r') as f:
                    lines = f.readlines()[1:]  # Skip header
                    for line in lines:
                        temp, label = line.strip().split(',')
                        temperatures.append(float(temp))
                        labels.append(label)
                temperatures = np.array(temperatures)
                labels = np.array(labels)
            
            print(f"Loaded {len(temperatures)} temperature readings from {csv_path}")
            return temperatures, labels
            
        except FileNotFoundError:
            print(f"CSV file {csv_path} not found. Using generated data instead.")
            return None, None
        except Exception as e:
            print(f"Error loading CSV: {e}. Using generated data instead.")
            return None, None
    


    @staticmethod
    def list_available_models():
        """List all available model files"""
        models_dir = Path("models")
        if not models_dir.exists():
            print("No models folder found.")
            return []
        
        all_files = list(models_dir.iterdir())
        
        # Find .keras files
        keras_files = [f for f in all_files if f.suffix == '.keras']
        
        # Find weight files (look for .weights.h5 files)
        weight_files = [f for f in all_files if f.name.endswith('.weights.h5')]
        
        # Get unique model names
        model_names = set()
        
        for keras_file in keras_files:
            base_name = keras_file.stem
            model_names.add(base_name)
        
        for weight_file in weight_files:
            if weight_file.name.endswith('.weights.h5'):
                base_name = weight_file.name.replace('.weights.h5', '')
                model_names.add(base_name)
        
        if model_names:
            print("Available models:")
            for i, model_name in enumerate(sorted(model_names), 1):
                print(f"  {i}. {model_name}")
                
                # Check for .keras file
                keras_path = models_dir / f"{model_name}.keras"
                if keras_path.exists():
                    size_mb = keras_path.stat().st_size / (1024 * 1024)
                    print(f"     📄 Complete model (.keras): {size_mb:.2f} MB")
                
                # Check for weights files
                weights_file = f"{model_name}.weights.h5"
                if any(f.name == weights_file for f in all_files):
                    print(f"     📄 Weights only: available")
                print()
        else:
            print("No saved models found in models folder.")
        
        return sorted(list(model_names))

    def simulate_sensor_demo(self):
        """Simulate putting a temperature sensor into different environments"""
        print("\n🌡️  Temperature Sensor Demo")
        print("=" * 50)
        print("Simulating sensor readings in different environments...")
        
        scenarios = [
            ("Room Temperature", [19, 20, 21, 20, 19]),
            ("Cold Water", [15, 10, 6, 4, 3]),
            ("Hot Water", [25, 30, 35, 38, 40]),
            ("Ice Water", [10, 5, 2, 1, 0]),
            ("Boiling Water", [30, 50, 70, 85, 95])
        ]
        
        for scenario_name, temps in scenarios:
            print(f"\n📍 {scenario_name}:")
            print(f"   Sensor readings: {temps}")
            
            if len(temps) >= self.sequence_length:
                prediction = self.predict(temps)
                if prediction:
                    pred = prediction[0]
                    print(f"   🔮 Prediction: {pred['class_name']} (confidence: {pred['confidence']:.1%})")
                    probs = pred['probabilities']
                    print(f"   📊 Probabilities - Cold: {probs['cold']:.1%}, Normal: {probs['normal']:.1%}, Hot: {probs['hot']:.1%}")
            else:
                print(f"   ⚠️  Need at least {self.sequence_length} readings for prediction")
            print("   " + "-" * 40)
        

def main():
    """Demo the temperature prediction model"""
    print("Temperature Prediction System")
    print("=" * 40)
    
    available_models = TemperaturePredictionModel.list_available_models()
    
    # Let user choose to train new model or load existing one
    if available_models:
        choice = input("\nTrain new model? (y/n): ").lower().strip()
        if choice == 'y':
            model_name = input("Enter model name (or press Enter for auto-generated): ").strip()
            model = TemperaturePredictionModel(model_name=model_name if model_name else None)
        else:
            # Use the most recent model (last in alphabetical order usually means most recent)
            latest_model = sorted(available_models)[-1]
            model = TemperaturePredictionModel(model_name=latest_model)
            if not model.load_model():
                print("Failed to load model, creating new one...")
                model = TemperaturePredictionModel()
    else:
        model = TemperaturePredictionModel()
    
    # Train or load model
    if not hasattr(model, 'model') or model.model is None:
        if not model.load_model():
            print("\nTraining new model...")
            temperatures = model.train_model()
            
            # Save model and weights
            print("\nSaving model...")
            model.save_model()  # Save complete model
            model.save_weights()  # Also save weights separately
            
            recent_temps = temperatures[-48:]  # Last 48 hours for demo
        else:
            recent_temps = model.generate_temperature_data(50)[-10:]  # Get last 10 readings for demo
    
    # Run sensor simulation demo
    model.simulate_sensor_demo()
    
    # Make predictions with sample data
    current_temp = recent_temps[-1] if len(recent_temps) > 0 else 20.0
    print(f"\nCurrent temperature: {current_temp:.1f}°C")
    print("\nReal-time predictions:")
    
    # Test with different sample sequences
    test_sequences = [
        [18, 19, 20, 21, 20],  # Room temp
        [20, 15, 10, 5, 3],    # Cooling down (cold water)
        [20, 25, 30, 35, 38]   # Heating up (hot water)
    ]
    
    for i, seq in enumerate(test_sequences, 1):
        pred = model.predict(seq)
        if pred:
            prediction = pred[0]
            print(f"Test {i} {seq}: {prediction['class_name']} "
                  f"(confidence: {prediction['confidence']:.1%})")
    
    # Run the sensor demo simulation
    model.simulate_sensor_demo()
       

if __name__ == "__main__":
    main()
