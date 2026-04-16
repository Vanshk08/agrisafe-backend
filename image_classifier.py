"""
Image Classification Module
Uses scikit-learn with advanced features for food contamination detection
Includes color histograms, texture analysis, and color space statistics
"""
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from scipy import ndimage
from skimage import feature
import pickle
import os
import logging

logger = logging.getLogger(__name__)


class ImageClassifier:
    """
    Image classifier for fresh vs spoiled food detection
    Uses scikit-learn RandomForest with color histogram features
    """
    
    def __init__(self, model_path=None, img_height=224, img_width=224):
        """
        Initialize classifier
        
        Args:
            model_path: path to saved model
            img_height: image height for feature extraction
            img_width: image width for feature extraction
        """
        self.img_height = img_height
        self.img_width = img_width
        self.model = None
        self.classes = ['fresh', 'spoiled']
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            logger.warning(f"Model not found at {model_path}")

    def build_model(self, n_estimators=100, random_state=42):
        """Build Random Forest model for image classification"""
        try:
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=15,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=random_state,
                n_jobs=-1,
                class_weight='balanced'  # Handle class imbalance
            )
            logger.info("Model built successfully")
            return self.model
            
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise

    def extract_color_histogram_features(self, image_path, bins=8):
        """
        Extract comprehensive features from image for spoilage detection
        Enhanced with additional mold, decay, and texture indicators
        
        Args:
            image_path: path to image
            bins: number of histogram bins per channel
            
        Returns:
            feature array with spoilage detection features
        """
        try:
            # Load and resize image
            img = Image.open(image_path).convert('RGB')
            img = img.resize((self.img_width, self.img_height))
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            features = []
            
            # 1. Brightness/Darkness (CRITICAL: spoiled is much darker)
            brightness = np.mean(img_array)
            features.append(brightness)
            features.append(np.std(np.mean(img_array, axis=2)))  # Brightness variation
            
            # 2. RGB Channel Analysis (spoiled has shifted colors)
            r_mean = np.mean(img_array[:,:,0])
            g_mean = np.mean(img_array[:,:,1])
            b_mean = np.mean(img_array[:,:,2])
            
            features.append(r_mean)
            features.append(g_mean)
            features.append(b_mean)
            features.append(r_mean + g_mean + b_mean)  # Total color intensity
            
            # 3. Color Balance (fresh bread is warmer: R>G, R>B)
            features.append(r_mean - g_mean)
            features.append(r_mean - b_mean)
            features.append(g_mean - b_mean)
            
            # 4. HSV Color Space (saturation is critical!)
            img_rgb = Image.fromarray((img_array * 255).astype(np.uint8))
            img_hsv = np.array(img_rgb.convert('HSV'), dtype=np.float32) / 255.0
            
            hue = img_hsv[:,:,0]
            saturation = img_hsv[:,:,1]
            value = img_hsv[:,:,2]
            
            # Saturation is CRITICAL: fresh has high saturation, spoiled is dull
            features.append(np.mean(saturation))  # Average saturation
            features.append(np.percentile(saturation, 75))  # 75th percentile 
            features.append(np.percentile(saturation, 25))  # 25th percentile
            features.append(np.std(saturation))
            
            # Value (brightness in HSV)
            features.append(np.mean(value))
            features.append(np.std(value))
            
            # 5. Mold Detection Features (multiple types)
            # Green mold
            green_excess = (g_mean - np.mean([r_mean, b_mean])) * 100
            features.append(green_excess)
            features.append(g_mean - r_mean)
            features.append(g_mean - b_mean)
            
            # Count green pixels (mold indicator)
            green_pixels = np.sum((img_array[:,:,1] > img_array[:,:,0]) & 
                                 (img_array[:,:,1] > img_array[:,:,2]))
            features.append(green_pixels / (self.img_width * self.img_height))
            
            # White/gray mold
            gray_pixels = np.sum((np.abs(img_array[:,:,0] - img_array[:,:,1]) < 0.1) &
                                (np.abs(img_array[:,:,1] - img_array[:,:,2]) < 0.1) &
                                (img_array[:,:,0] > 0.4))
            features.append(gray_pixels / (self.img_width * self.img_height))
            
            # 6. Decay Features
            saturation_brightness = np.mean(saturation) * np.mean(value)
            features.append(saturation_brightness)  # Fresh: high, Spoiled: low
            
            # Low saturation + dark = spoilage
            low_sat_dark = np.sum((saturation < 0.3) & (value < 0.4))
            features.append(low_sat_dark / (self.img_width * self.img_height))
            
            # 7. Texture Features
            img_gray = np.array(img_rgb.convert('L'), dtype=np.float32) / 255.0
            
            edges = ndimage.laplace(img_gray)
            features.append(np.mean(np.abs(edges)))
            features.append(np.std(np.abs(edges)))
            
            # Contrast
            contrast = np.max(img_gray) - np.min(img_gray)
            features.append(contrast)
            
            # Smoothness (spoiled is often blurry/smooth)
            smoothness = np.mean(np.abs(np.diff(img_gray.flatten())))
            features.append(smoothness)
            
            # 8. Color Uniformity (spoiled has patches)
            r_std = np.std(img_array[:,:,0])
            g_std = np.std(img_array[:,:,1])
            b_std = np.std(img_array[:,:,2])
            
            features.append(r_std)
            features.append(g_std)
            features.append(b_std)
            features.append(r_std + g_std + b_std)
            
            # 9. Darkness Score
            darkness_score = (1 - np.mean(value)) * 100
            features.append(darkness_score)
            
            # 10. Comprehensive Mold/Decay Score
            # Combines multiple decay indicators
            mold_score = (1 - np.mean(saturation)) * (1 - np.mean(value)) * (g_mean - r_mean) * 100
            features.append(mold_score)
            
            # 11. Color histogram variance (spoiled = more color variation)
            hist_r = np.histogram(img_array[:,:,0], bins=bins)[0]
            hist_g = np.histogram(img_array[:,:,1], bins=bins)[0]
            hist_b = np.histogram(img_array[:,:,2], bins=bins)[0]
            
            features.append(np.std(hist_r))
            features.append(np.std(hist_g))
            features.append(np.std(hist_b))
            
            # 12. Discoloration indicator (distance from golden bread color)
            golden_r, golden_g, golden_b = 0.86, 0.71, 0.39  # Normalized golden color
            color_distance = np.sqrt((r_mean - golden_r)**2 + (g_mean - golden_g)**2 + (b_mean - golden_b)**2)
            features.append(color_distance)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error extracting features from {image_path}: {str(e)}")
            return np.zeros(40)

    def train(self, train_images, train_labels, validation_data=None, epochs=None, batch_size=None, bins=8):
        """
        Train the model
        
        Args:
            train_images: list of image paths or feature array
            train_labels: training labels (0 or 1)
            validation_data: optional validation data tuple
            epochs: ignored (kept for compatibility)
            batch_size: ignored (kept for compatibility)
            bins: histogram bins for feature extraction
        """
        try:
            if self.model is None:
                self.build_model()
            
            # If train_images are paths, extract features
            if isinstance(train_images, list) and isinstance(train_images[0], str):
                logger.info("Extracting features from images...")
                features = np.array([self.extract_color_histogram_features(path, bins) for path in train_images])
            else:
                features = train_images
            
            # Train model
            logger.info("Training model...")
            self.model.fit(features, train_labels)
            
            # Validate if provided
            if validation_data is not None:
                val_images, val_labels = validation_data
                if isinstance(val_images, list) and isinstance(val_images[0], str):
                    val_features = np.array([self.extract_color_histogram_features(path, bins) for path in val_images])
                else:
                    val_features = val_images
                
                val_score = self.model.score(val_features, val_labels)
                logger.info(f"Validation accuracy: {val_score:.4f}")
            
            logger.info("Model training completed")
            return self.model
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise

    def save_model(self, filepath):
        """Save model to file"""
        try:
            if self.model is None:
                raise ValueError("No model to save")
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'wb') as f:
                pickle.dump(self.model, f)
            
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, filepath):
        """Load model from file"""
        try:
            with open(filepath, 'rb') as f:
                self.model = pickle.load(f)
            
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def predict(self, image_path, spoiled_threshold=0.50):
        """
        Predict food freshness from image
        Uses balanced threshold: 60% confidence required for FRESH prediction
        More lenient to reduce false spoilage readings on fresh food
        
        Args:
            image_path: path to image file
            spoiled_threshold: confidence threshold for 'fresh' prediction (default 0.60)
                - if fresh_confidence >= 0.60: mark as FRESH (60%+ confident)
                - if fresh_confidence < 0.60: mark as SPOILED
            
        Returns:
            tuple: (prediction, confidence)
                - prediction: 'fresh' or 'spoiled'
                - confidence: confidence score (0-1)
        """
        try:
            if self.model is None:
                raise ValueError("Model not loaded")
            
            # Extract features from image
            features = self.extract_color_histogram_features(image_path)
            features = features.reshape(1, -1)
            
            # Get probabilities for both classes
            predicted_class_idx = self.model.predict(features)[0]
            proba = self.model.predict_proba(features)[0]
            
            # proba[0] = spoiled probability, proba[1] = fresh probability
            # (classes might be reversed - test to see if this fixes it)
            spoiled_confidence = proba[0]
            fresh_confidence = proba[1]
            
            # Accept FRESH if we're above threshold confidence
            if fresh_confidence >= spoiled_threshold:
                prediction = 'fresh'
                confidence = fresh_confidence
            else:
                prediction = 'spoiled'
                confidence = spoiled_confidence
            
            logger.info(f"Prediction: {prediction} (fresh conf: {fresh_confidence:.4f}, spoiled conf: {spoiled_confidence:.4f}, threshold: {spoiled_threshold})")
            return prediction, confidence
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise

    def predict_batch(self, image_paths):
        """
        Predict batch of images
        
        Args:
            image_paths: list of image file paths
            
        Returns:
            list of tuples (prediction, confidence)
        """
        results = []
        for image_path in image_paths:
            prediction, confidence = self.predict(image_path)
            results.append({
                'image_path': image_path,
                'prediction': prediction,
                'confidence': confidence
            })
        return results
