#!/usr/bin/env python3
"""Inference script for single image prediction."""

import argparse
import os
import sys
import json
import torch
from pathlib import Path
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.transforms import RGBTransform, FrequencyTransform, NoiseTransform
from src.models.detector import load_detector


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on single image")
    
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Image size for model input",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file (optional)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output",
    )
    
    return parser.parse_args()


class ImagePredictor:
    """Single image prediction wrapper."""
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        image_size: int = 256,
        threshold: float = 0.5,
    ):
        self.device = device
        self.image_size = image_size
        self.threshold = float(threshold)
        
        # Load model
        self.model = load_detector(checkpoint_path, device=device)
        self.model.eval()
        
        # Initialize transforms
        self.rgb_transform = RGBTransform(size=image_size)
        self.freq_transform = FrequencyTransform(size=image_size)
        self.noise_transform = NoiseTransform(size=image_size)
    
    def predict(self, image_path: str) -> dict:
        """Predict whether an image is real or AI-generated.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dict with prediction results
        """
        # Load image
        img = Image.open(image_path).convert('RGB')
        original_size = img.size
        
        # Extract features
        rgb = self.rgb_transform(img).unsqueeze(0).to(self.device)
        freq = self.freq_transform(img).unsqueeze(0).to(self.device)
        noise = self.noise_transform(img).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            _, probs = self.model.predict(rgb, freq, noise)
        
        prob_real = probs[0, 0].item()
        prob_fake = probs[0, 1].item()
        pred_label = 1 if prob_fake >= self.threshold else 0
        
        # Determine class name and confidence
        if pred_label == 0:
            class_name = "REAL"
            confidence = prob_real
        else:
            class_name = "FAKE (AI-Generated)"
            confidence = prob_fake
        
        return {
            'image_path': str(image_path),
            'original_size': original_size,
            'prediction': class_name,
            'predicted_label': pred_label,
            'confidence': confidence,
            'probabilities': {
                'real': prob_real,
                'fake': prob_fake,
            },
        }
    
    def predict_batch(self, image_paths: list) -> list:
        """Predict on multiple images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of prediction dicts
        """
        results = []
        for path in image_paths:
            try:
                result = self.predict(path)
                results.append(result)
            except Exception as e:
                results.append({
                    'image_path': str(path),
                    'error': str(e),
                })
        return results


def main():
    args = parse_args()
    
    # Validate input
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    # Determine device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        device = "cpu"
    
    # Create predictor
    print(f"Loading model from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    threshold = float(checkpoint.get("best_threshold", 0.5))
    print(f"Using decision threshold: {threshold:.3f}")
    predictor = ImagePredictor(
        checkpoint_path=args.checkpoint,
        device=device,
        image_size=args.image_size,
        threshold=threshold,
    )
    
    # Run prediction
    print(f"Analyzing: {args.image}")
    result = predictor.predict(args.image)
    
    # Print results
    print("\n" + "=" * 50)
    print("PREDICTION RESULT")
    print("=" * 50)
    print(f"Image: {result['image_path']}")
    print(f"Size: {result['original_size'][0]}x{result['original_size'][1]}")
    print(f"\nPrediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.1%}")
    
    if args.verbose:
        print(f"\nProbabilities:")
        print(f"  Real: {result['probabilities']['real']:.4f}")
        print(f"  Fake: {result['probabilities']['fake']:.4f}")
    
    print("=" * 50)
    
    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    # Exit with code based on prediction
    # 0 = real, 1 = fake
    sys.exit(result['predicted_label'])


if __name__ == "__main__":
    main()
