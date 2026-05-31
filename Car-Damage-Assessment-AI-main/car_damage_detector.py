import cv2
import numpy as np
from PIL import Image
import torch
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional, Union
import os
import urllib.request
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CarDamageDetector:
    """
    Car damage detection system using YOLOv8 model.
    
    This class handles the detection and classification of various types
    of car damage including scratches, dents, paint damage, and broken parts.
    """
    
    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.5):
        """
        Initialize the car damage detector.
        
        Args:
            model_path (str, optional): Path to custom trained model
            confidence_threshold (float): Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        self.model_path = model_path or "runs/detect/car_damage_model/weights/best.pt"
        self.device = self._get_device()
        self.model = None
        self.use_mock = False
        self.class_names = {
            0: "dent",
            1: "scratch",
            2: "crack",
            3: "glass shatter",
            4: "lamp broken",
            5: "tire flat"
        }
        
        # Damage severity mapping based on area and type
        self.severity_mapping = {
            "scratch": {"light": (0, 5), "moderate": (5, 15), "severe": (15, 100)},
            "dent": {"light": (0, 3), "moderate": (3, 10), "severe": (10, 100)},
            "crack": {"light": (0, 2), "moderate": (2, 8), "severe": (8, 100)},
            "glass shatter": {"light": (0, 2), "moderate": (2, 8), "severe": (8, 100)},
            "lamp broken": {"light": (0, 2), "moderate": (2, 5), "severe": (5, 100)},
            "tire flat": {"light": (0, 5), "moderate": (5, 15), "severe": (15, 100)}
        }

        # Cost estimation per damage type (base costs in USD)
        self.cost_estimates = {
            "scratch": {"light": 100, "moderate": 300, "severe": 800},
            "dent": {"light": 200, "moderate": 500, "severe": 1200},
            "crack": {"light": 250, "moderate": 600, "severe": 1500},
            "glass shatter": {"light": 150, "moderate": 400, "severe": 900},
            "lamp broken": {"light": 100, "moderate": 250, "severe": 600},
            "tire flat": {"light": 50, "moderate": 100, "severe": 300}
        }
        
        self._load_model()
    
    def _get_device(self) -> str:
        """Determine the best available device for inference."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _load_model(self) -> None:
        """Load the YOLO model for damage detection."""
        try:
            # Check if custom model exists
            if os.path.exists(self.model_path):
                logger.info(f"Loading custom trained model from {self.model_path}")
                self.model = YOLO(self.model_path)
            else:
                logger.warning(f"Trained model not found at {self.model_path}. Falling back to mock detection mode.")
                self.use_mock = True
                return
            
            # Move model to appropriate device
            if self.device != "cpu":
                self.model.to(self.device)
                
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}. Falling back to mock detection mode.")
            self.use_mock = True
    
    def _preprocess_image(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Preprocess image for model inference.
        
        Args:
            image: Input image (path, numpy array, or PIL Image)
            
        Returns:
            np.ndarray: Preprocessed image array
        """
        if isinstance(image, str):
            # Load from file path
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Could not load image from {image}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            # Convert PIL Image to numpy array
            img = np.array(image)
        elif isinstance(image, np.ndarray):
            # Use numpy array directly
            img = image.copy()
        else:
            raise TypeError("Image must be a file path, PIL Image, or numpy array")
        
        return img
    
    def _calculate_damage_area(self, bbox: List[int], image_shape: Tuple[int, int]) -> float:
        """
        Calculate the percentage of image area covered by damage.
        
        Args:
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            image_shape: Image dimensions (height, width)
            
        Returns:
            float: Percentage of image area covered by damage
        """
        x1, y1, x2, y2 = bbox
        damage_area = (x2 - x1) * (y2 - y1)
        total_area = image_shape[0] * image_shape[1]
        return (damage_area / total_area) * 100
    
    def _classify_severity(self, damage_type: str, area_percentage: float) -> str:
        """
        Classify damage severity based on type and area coverage.
        
        Args:
            damage_type: Type of damage detected
            area_percentage: Percentage of image area covered
            
        Returns:
            str: Severity level ('light', 'moderate', 'severe')
        """
        if damage_type not in self.severity_mapping:
            return "moderate"  # Default classification
        
        thresholds = self.severity_mapping[damage_type]
        
        if area_percentage <= thresholds["light"][1]:
            return "light"
        elif area_percentage <= thresholds["moderate"][1]:
            return "moderate"
        else:
            return "severe"
    
    def _estimate_repair_cost(self, damage_type: str, severity: str, area_percentage: float) -> int:
        """
        Estimate repair cost based on damage type and severity.
        
        Args:
            damage_type: Type of damage
            severity: Severity level
            area_percentage: Area coverage percentage
            
        Returns:
            int: Estimated repair cost in USD
        """
        if damage_type not in self.cost_estimates:
            return 0
        
        base_cost = self.cost_estimates[damage_type][severity]
        
        # Apply area multiplier for larger damages
        area_multiplier = max(1.0, area_percentage / 10.0)
        estimated_cost = int(base_cost * area_multiplier)
        
        return estimated_cost
    
    def detect_damage(self, image: Union[str, np.ndarray, Image.Image]) -> Dict:
        """
        Detect damage in a car image.
        
        Args:
            image: Input image (path, numpy array, or PIL Image)
            
        Returns:
            Dict: Detection results containing damages and metadata
        """
        try:
            # Preprocess image
            img_array = self._preprocess_image(image)
            original_shape = img_array.shape[:2]  # (height, width)
            
            # Mock mode: generate simulated detections
            if self.use_mock:
                return self._mock_detect(img_array, original_shape)
            
            # Run inference
            results = self.model(img_array, conf=self.confidence_threshold, verbose=False)
            
            # Process results
            detections = []
            
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                
                for i in range(len(boxes)):
                    # Extract bounding box coordinates
                    bbox = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
                    confidence = float(boxes.conf[i].cpu().numpy())
                    class_id = int(boxes.cls[i].cpu().numpy())
                    
                    # Use real class prediction from trained model
                    damage_type = self.class_names.get(class_id, "unknown")
                    
                    # Calculate damage area percentage
                    area_percentage = self._calculate_damage_area(bbox, original_shape)
                    
                    # Classify severity
                    severity = self._classify_severity(damage_type, area_percentage)
                    
                    # Estimate repair cost
                    estimated_cost = self._estimate_repair_cost(damage_type, severity, area_percentage)
                    
                    detection = {
                        "type": damage_type,
                        "severity": severity,
                        "confidence": confidence,
                        "bbox": bbox,
                        "area_percentage": round(area_percentage, 2),
                        "estimated_cost": estimated_cost,
                        "location": self._describe_location(bbox, original_shape)
                    }
                    
                    detections.append(detection)
            
            # Create result summary
            result = {
                "image_shape": original_shape,
                "total_damages": len(detections),
                "damages": detections,
                "total_estimated_cost": sum([d["estimated_cost"] for d in detections]),
                "highest_severity": self._get_highest_severity(detections),
                "processing_info": {
                    "model_used": "YOLOv8",
                    "device": self.device,
                    "confidence_threshold": self.confidence_threshold
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error during damage detection: {str(e)}")
            raise RuntimeError(f"Damage detection failed: {str(e)}")

    def _mock_detect(self, img_array: np.ndarray, original_shape: Tuple[int, int]) -> Dict:
        """Generate simulated detections for demo without trained model."""
        import random
        random.seed(42)
        h, w = original_shape
        detections = []
        damage_types = ["dent", "scratch", "crack"]
        for _ in range(random.randint(1, 3)):
            bx = random.randint(w // 4, 3 * w // 4)
            by = random.randint(h // 4, 3 * h // 4)
            bw = random.randint(40, 120)
            bh = random.randint(30, 100)
            bbox = [max(0, bx - bw // 2), max(0, by - bh // 2), min(w, bx + bw // 2), min(h, by + bh // 2)]
            dmg = random.choice(damage_types)
            area_pct = self._calculate_damage_area(bbox, original_shape)
            sev = self._classify_severity(dmg, area_pct)
            detections.append({
                "type": dmg,
                "severity": sev,
                "confidence": round(random.uniform(0.65, 0.92), 2),
                "bbox": bbox,
                "area_percentage": round(area_pct, 2),
                "estimated_cost": self._estimate_repair_cost(dmg, sev, area_pct),
                "location": self._describe_location(bbox, original_shape)
            })
        return {
            "image_shape": original_shape,
            "total_damages": len(detections),
            "damages": detections,
            "total_estimated_cost": sum(d["estimated_cost"] for d in detections),
            "highest_severity": self._get_highest_severity(detections),
            "processing_info": {
                "model_used": "YOLOv8 (mock mode - train for real inference)",
                "device": self.device,
                "confidence_threshold": self.confidence_threshold
            }
        }
    
    def _simulate_damage_classification(self, bbox: List[int], image: np.ndarray) -> str:
        """
        Simulate damage type classification for demo purposes.
        In production, this would be handled by the trained model.
        
        Args:
            bbox: Bounding box coordinates
            image: Image array
            
        Returns:
            str: Simulated damage type
        """
        # Simple simulation based on bounding box characteristics
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = width / height if height > 0 else 1
        
        # Simulate different damage types based on shape characteristics
        if aspect_ratio > 2.0:
            return "scratch"
        elif width * height < 5000:  # Small area
            return "paint_damage"
        elif aspect_ratio < 0.8:
            return "dent"
        else:
            return "broken_part"
    
    def _describe_location(self, bbox: List[int], image_shape: Tuple[int, int]) -> str:
        """
        Describe the location of damage on the vehicle.
        
        Args:
            bbox: Bounding box coordinates
            image_shape: Image dimensions
            
        Returns:
            str: Human-readable location description
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        height, width = image_shape
        
        # Determine horizontal position
        if center_x < width / 3:
            horizontal = "left"
        elif center_x > 2 * width / 3:
            horizontal = "right"
        else:
            horizontal = "center"
        
        # Determine vertical position
        if center_y < height / 3:
            vertical = "upper"
        elif center_y > 2 * height / 3:
            vertical = "lower"
        else:
            vertical = "middle"
        
        return f"{vertical} {horizontal}"
    
    def _get_highest_severity(self, detections: List[Dict]) -> str:
        """
        Determine the highest severity level among all detections.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            str: Highest severity level
        """
        if not detections:
            return "none"
        
        severity_levels = {"light": 1, "moderate": 2, "severe": 3}
        max_severity_value = max([severity_levels.get(d["severity"], 0) for d in detections])
        
        for severity, value in severity_levels.items():
            if value == max_severity_value:
                return severity
        
        return "light"
    
    def annotate_image(self, image: Union[str, np.ndarray, Image.Image], 
                      detections: List[Dict]) -> np.ndarray:
        """
        Annotate image with detection results.
        
        Args:
            image: Input image
            detections: List of detection results
            
        Returns:
            np.ndarray: Annotated image
        """
        img_array = self._preprocess_image(image)
        annotated_img = img_array.copy()
        
        # Color mapping for different damage types
        colors = {
            "scratch": (0, 255, 0),       # Green
            "dent": (255, 165, 0),        # Orange
            "crack": (255, 0, 0),         # Red
            "glass shatter": (255, 0, 255), # Magenta
            "lamp broken": (0, 255, 255),  # Cyan
            "tire flat": (128, 0, 128)     # Purple
        }
        
        for detection in detections:
            bbox = detection["bbox"]
            damage_type = detection["type"]
            confidence = detection["confidence"]
            severity = detection["severity"]
            
            x1, y1, x2, y2 = bbox
            color = colors.get(damage_type, (255, 255, 0))
            
            # Draw bounding box
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            label = f"{damage_type.replace('_', ' ').title()}"
            sublabel = f"{severity.capitalize()} ({confidence:.2f})"
            
            # Calculate text size and position
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
            (sub_width, sub_height), _ = cv2.getTextSize(sublabel, font, font_scale-0.1, thickness-1)
            
            # Draw label background
            cv2.rectangle(annotated_img, 
                         (x1, y1 - text_height - sub_height - 10), 
                         (x1 + max(text_width, sub_width), y1), 
                         color, -1)
            
            # Draw label text
            cv2.putText(annotated_img, label, (x1, y1 - sub_height - 5), 
                       font, font_scale, (255, 255, 255), thickness)
            cv2.putText(annotated_img, sublabel, (x1, y1 - 2), 
                       font, font_scale-0.1, (255, 255, 255), thickness-1)
        
        return annotated_img
    
    def batch_detect(self, image_paths: List[str]) -> List[Dict]:
        """
        Process multiple images for damage detection.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List[Dict]: Detection results for each image
        """
        results = []
        
        for i, image_path in enumerate(image_paths):
            try:
                logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
                result = self.detect_damage(image_path)
                result["image_path"] = image_path
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {image_path}: {str(e)}")
                results.append({
                    "image_path": image_path,
                    "error": str(e),
                    "total_damages": 0,
                    "damages": []
                })
        
        return results
    
    def update_confidence_threshold(self, new_threshold: float) -> None:
        """
        Update the confidence threshold for detections.
        
        Args:
            new_threshold: New confidence threshold (0.0 to 1.0)
        """
        if 0.0 <= new_threshold <= 1.0:
            self.confidence_threshold = new_threshold
            logger.info(f"Confidence threshold updated to {new_threshold}")
        else:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")

def main():
    """
    Example usage of the CarDamageDetector class.
    """
    # Initialize detector
    detector = CarDamageDetector(confidence_threshold=0.5)
    
    # Example with a sample image (would need actual image file)
    # results = detector.detect_damage("sample_car_damage.jpg")
    # print(f"Detected {results['total_damages']} damage areas")
    
    print("CarDamageDetector initialized successfully")
    print(f"Device: {detector.device}")
    print(f"Model: {detector.model_path}")
    print(f"Confidence threshold: {detector.confidence_threshold}")

if __name__ == "__main__":
    main()
