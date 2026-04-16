#!/usr/bin/env python
"""Debug script to analyze predictions"""
import os
import sys
import numpy as np
from PIL import Image
from image_classifier import ImageClassifier

classifier = ImageClassifier('../models/saved_models/food_classifier.pkl')

# List available images
uploads_dir = 'uploads'
if os.path.exists(uploads_dir):
    images = [f for f in os.listdir(uploads_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    print(f"Found {len(images)} images in uploads/")
    print(images)
    
    if images:
        test_image_path = os.path.join(uploads_dir, images[0])
        print("\n" + "="*60)
        print(f"Analyzing: {test_image_path}")
        print("="*60)
        
        try:
            features = classifier.extract_color_histogram_features(test_image_path)
            print(f"\nFeatures extracted: {len(features)}")
            print(f"\nKey indicators:")
            print(f"  Brightness: {features[0]:.3f} (0=dark, 1=bright)")
            print(f"  RGB: R={features[2]:.3f}, G={features[3]:.3f}, B={features[4]:.3f}")
            print(f"  Saturation: {features[10]:.3f} (fresh=high, spoiled=low)")
            print(f"  HSV Value (brightness): {features[14]:.3f}")
            print(f"  Darkness score: {features[-2]:.1f} (spoiled=high)")
            print(f"  Mold score: {features[-1]:.3f}")
            
            if classifier.model:
                pred, conf = classifier.predict(test_image_path)
                print(f"\nPrediction: {pred.upper()}")
                print(f"Confidence: {conf*100:.2f}%")
                
                features_array = features.reshape(1, -1)
                proba = classifier.model.predict_proba(features_array)
                print(f"\nProbabilities:")
                print(f"  Fresh: {proba[0][0]*100:.2f}%")
                print(f"  Spoiled: {proba[0][1]*100:.2f}%")
            else:
                print("Model not loaded")
                
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
else:
    print(f"No uploads directory found")
