import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = "C:\\Users\\Yousuf Yasser Rabie\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe"
from PIL import Image
import torch
import torchvision.transforms as transforms
from transformers import pipeline, AutoTokenizer, AutoModel
import re
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import base64
import io
from database import VerificationService

class ProofVerifier:
    """
    AI-powered proof verification system for student activities.
    Handles both text extraction from certificates and image analysis for activity photos.
    """
    
    def __init__(self, db_path: str = "verification_data.db"):
        self.setup_logging()
        self.initialize_models()
        self.define_keywords()
        self.verification_service = VerificationService(db_path)
        
    def setup_logging(self):
        """Setup logging for verification process"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('proof_verification.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def initialize_models(self):
        """Initialize AI models for text and image analysis"""
        try:
            # Initialize CLIP model for image-text matching
            self.clip_model = pipeline("zero-shot-image-classification", 
                                     model="openai/clip-vit-base-patch32")
            
            # Initialize text classifier for document analysis
            self.text_classifier = pipeline("zero-shot-classification",
                                           model="facebook/bart-large-mnli")
            
            # Initialize object detection model
            self.object_detector = pipeline("object-detection",
                                          model="facebook/detr-resnet-50")
            
        
            
            # Activity detection model (image classification)
            self.activity_classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
            self.logger.info("Models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing models: {str(e)}")
            raise
    
    def define_keywords(self):
        """Define keywords for different activity categories"""
        self.category_keywords = {
            'courses': [
                'certificate', 'certification', 'awarded', 'completed', 'achievement',
                'course', 'completion', 'successfully', 'graduated', 'diploma',
                'program', 'training', 'education', 'learning', 'study', 'bootcamp'
            ],
            
            'workshops': [
                'certificate', 'certification', 'awarded', 'completed', 'achievement',
                'workshop', 'training', 'session', 'participant', 'attendance',
                'conducted by', 'organized by', 'facilitated by', 'work id',
                'employee id', 'staff id', 'project', 'team member', 'role',
                'position', 'department', 'seminar', 'tutorial', 'hands-on', 'bootcamp'
            ],
            
            'student_activities_aisec': [
                'AISEC','Enactus','IEEE','member', 'participant', 'team', 'work id','society'
                'project', 'initiative', 'event', 'activity','leadership',
                'student id', 'position', 'role', 'committee', 'club', 
                'student organization', 
            ],
            
            'other_student_activities': [
                'tree', 'msp', 'student partner', 'member', 'participant',
                'team', 'project', 'initiative', 'event', 'activity', 'work id',
                'student id', 'position', 'role', 'committee', 'organization'
            ],
            
            'volunteering_cop': [
                'volunteer', 'volunteering', 'community service', 'contribution',
                'appreciation', 'recognition', 'thank you', 'project', 'initiative',
                'team', 'role', 'position', 'impact', 'community', 'cop',
                'social responsibility', 'service'
            ],
            
            'volunteering_icareer': [
                'volunteer', 'volunteering', 'icareer', 'career', 'mentorship',
                'guidance', 'support', 'contribution', 'appreciation', 'recognition',
                'team', 'role', 'position', 'impact', 'community'
            ],
            
            'youtube': [
                'youtube', 'channel', 'subscriber', 'views', 'content creator',
                'video', 'upload', 'monetization', 'analytics', 'creator',
                'influencer', 'social media'
            ],
            
            'internship_cib': [
                'internship', 'intern', 'cib', 'training period', 'work experience',
                'company', 'organization', 'department', 'work id', 'employee id',
                'project', 'team', 'role', 'position', 'supervisor', 'mentor','bank'
            ],
            
            'real_internship': [
                'internship', 'intern', 'dsquares', 'training period', 'work experience',
                'company', 'organization', 'department', 'work id', 'employee id',
                'project', 'team', 'role', 'position', 'supervisor', 'mentor',
                'employment', 'professional'
            ],
            
            'sports': [
                'sports', 'team', 'player', 'athlete', 'training', 'practice',
                'competition', 'match', 'tournament', 'coach', 'uniform',
                'equipment', 'facility', 'stadium', 'field', 'court',
                'fitness', 'exercise', 'physical activity'
            ],
            
            'social': [
                'social', 'event', 'gathering', 'meeting', 'community',
                'networking', 'celebration', 'party', 'cultural', 'festival',
                'social activity', 'interaction', 'people'
            ],
            
            'arts': [
                'art', 'exhibition', 'performance', 'show', 'display',
                'gallery', 'studio', 'work', 'piece', 'creation', 'artist',
                'exhibit', 'display', 'venue', 'stage', 'creative',
                'painting', 'music', 'theater', 'dance'
            ],
            
            'time_spent': [
                'quran', 'religious', 'prayer', 'meditation','worship',
                'study', 'learning', 'reading', 'contemplation', 
                'islamic', 'faith', 'devotion'
            ]
        }
        
    
    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from image using OCR with enhanced preprocessing"""
        try:
            # Read the image
            image_cv = cv2.imread(image_path)
            if image_cv is None:
                self.logger.error("Failed to read image")
                return ""

            # Convert to grayscale
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            
            # Apply multiple preprocessing techniques
            # 1. Basic denoising
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # 2. Adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # 3. Dilation to make text more prominent
            kernel = np.ones((1,1), np.uint8)
            dilated = cv2.dilate(thresh, kernel, iterations=1)
            
            # 4. Additional sharpening
            kernel_sharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(dilated, -1, kernel_sharp)
            
            # Try multiple OCR configurations and combine results
            texts = []
            
            # Config 1: Standard
            text1 = pytesseract.image_to_string(sharpened, config='--psm 6')
            texts.append(text1)
            
            # Config 2: With orientation and script detection
            text2 = pytesseract.image_to_string(sharpened, config='--psm 3')
            texts.append(text2)
            
            # Config 3: Treat as single uniform block of text
            text3 = pytesseract.image_to_string(sharpened, config='--psm 4')
            texts.append(text3)
            
            # Combine all extracted texts
            combined_text = ' '.join(texts)
            
            # Clean up the text
            # Remove extra whitespace and normalize
            cleaned_text = ' '.join(combined_text.split())
            
            self.logger.info(f"Extracted text length: {len(cleaned_text)} characters")
            self.logger.info(f"Full extracted text: {cleaned_text}")
            
            return cleaned_text
            
        except Exception as e:
            self.logger.error(f"Error extracting text from image: {str(e)}")
            return ""
        
    def verify_text_keywords(self, text: str, category: str, activity_title: str) -> Dict:
        """Verify if text contains relevant keywords for the category"""
        keywords = self.category_keywords.get(category, [])
        text_lower = text.lower()
        
        found_keywords = []
        for keyword in keywords:
            if keyword.lower() in text_lower:
                found_keywords.append(keyword)
        
        # Calculate relevance score
        relevance_score = min(len(found_keywords) * 0.3, 1.0) if keywords else 0
        
        # Use zero-shot classification for additional verification
        try:
            classification_labels = [
                f"{category} and {activity_title} related document",
                f"certificate or award of {activity_title} and {category}",
                f"{category} and {activity_title} official document",
                "irrelevant document",
                "fake documents",
                "fake certificate or award"
            ]
            
            classification_result = self.text_classifier(text, classification_labels)
            top_label = classification_result['labels'][0]
            top_score = classification_result['scores'][0]
            
        except:
            top_label = "unknown"
            top_score = 0.0
        
        return {
            'found_keywords': found_keywords,
            'keyword_count': len(found_keywords),
            'relevance_score': relevance_score,
            'classification': top_label,
            'classification_score': top_score,
            'is_relevant': relevance_score > 0.5 or (top_score > 0.5 and category in top_label)
        }
    
    def _check_activity_name_in_text(self, text: str, activity_title: str) -> Dict:
        """
        Check if activity name (course/workshop) appears in the certificate text.
        Returns a dict with match status and matched words.
        """
        if not activity_title or not text:
            return {'found': False, 'matched_words': [], 'score': 0.0}
            
        # Convert both to lower case for case-insensitive matching
        text = text.lower()
        activity_words = [word.lower() for word in activity_title.split() 
                         if len(word) > 2]  # Ignore very short words
        
        # Remove common words that might cause false positives
        stop_words = {'the', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
        activity_words = [word for word in activity_words if word not in stop_words]
        
        # Find matching words
        matched_words = [word for word in activity_words if word in text]
        
        # Calculate match score based on number of matched words
        score = len(matched_words) / len(activity_words) if activity_words else 0.0
        
        return {
            'found': len(matched_words) > 0,
            'matched_words': matched_words,
            'score': score
        }
    
    def _check_name_in_text(self, text: str, full_name: str) -> Dict:
        """
        Check if user's name appears in the certificate text with improved matching.
        """
        self.logger.info(f"Checking name in text. Name to check: '{full_name}'")
        
        if not full_name or not text:
            self.logger.warning(f"Missing input - Name: '{full_name}', Text empty: {not text}")
            return {'found': False, 'matched_parts': [], 'score': 0.0}
            
        # Convert both to lower case and normalize spaces
        text = ' '.join(text.lower().split())
        full_name = ' '.join(full_name.lower().split())
        
        # Split name into parts (first name, middle name, last name)
        name_parts = [part for part in full_name.split() if len(part) > 1]  # Ignore initials
        
        self.logger.info(f"Name parts to check: {name_parts}")
        
        # Common name prefixes/suffixes to ignore
        ignore_parts = {'mr', 'mrs', 'ms', 'dr', 'prof', 'jr', 'sr', 'ii', 'iii', 'iv'}
        name_parts = [part for part in name_parts if part not in ignore_parts]
        
        # Find matching name parts with fuzzy matching
        matched_parts = []
        for part in name_parts:
            # Try exact match first
            if part in text:
                matched_parts.append(part)
                self.logger.info(f"Found exact name part in text: {part}")
                continue
                
            # Try with common character substitutions
            substitutions = {
                'o': '0',
                'i': '1',
                'l': '1',
                's': '5',
                'a': '@',
                'e': '3'
            }
            
            # Create variations of the name part
            variations = [part]
            for char, sub in substitutions.items():
                if char in part:
                    variations.append(part.replace(char, sub))
                if sub in part:
                    variations.append(part.replace(sub, char))
            
            # Check each variation
            for variation in variations:
                if variation in text:
                    matched_parts.append(part)  # Add original part
                    self.logger.info(f"Found name part variation in text: {variation} for {part}")
                    break
            else:
                self.logger.info(f"Name part not found in text (including variations): {part}")
        
        # Calculate score based on matched parts
        score = 0.0
        if matched_parts:
            # Try to find full name with flexible spacing
            full_name_parts = ' '.join(name_parts)
            if full_name_parts in text or any(
                full_name_parts.replace(' ', sep) in text 
                for sep in ['.', '-', '_', '']
            ):
                score = 1.0
                self.logger.info("Found full name match!")
            elif name_parts[0] in matched_parts:  # First name match
                score = 0.7
                self.logger.info("Found first name match")
            else:  # Partial matches
                score = len(matched_parts) * 0.3
                self.logger.info(f"Found partial matches, score: {score}")
            score = min(score, 1.0)  # Cap at 1.0
        
        result = {
            'found': len(matched_parts) > 0,
            'matched_parts': matched_parts,
            'score': score,
            'first_name_found': name_parts[0] in matched_parts if name_parts else False
        }
        
        self.logger.info(f"Name check result: {result}")
        return result
    
    def _is_proof_valid_for_time(self, verification_result: Dict) -> Dict:
        """
        Check if a proof is valid for time counting based on verification status,
        name matching, and activity title matching.
        Returns a dict with validity status and reason.
        """
        # Default response
        response = {
            'is_valid': False,
            'reason': None,
            'confidence_percentage': 0
        }

        # Get verification details
        details = verification_result.get('details', {})
        name_check = details.get('name_check', {})
        activity_check = details.get('activity_name_check', {})
        
        # Calculate overall confidence percentage
        confidence_factors = []
        
        # Base verification confidence
        if verification_result['verification_status'] == 'verified':
            confidence_factors.append(1.0)
        elif verification_result['verification_status'] == 'partially_verified':
            confidence_factors.append(0.5)
        else:
            response['reason'] = "Proof not verified"
            return response
        
        # Name check confidence
        if name_check:
            name_score = name_check.get('score', 0)
            confidence_factors.append(name_score)
            if not name_check.get('found', False):
                response['reason'] = "Name not found in certificate"
                return response
        
        # Activity title check confidence (for courses and workshops)
        if activity_check:
            activity_score = activity_check.get('score', 0)
            confidence_factors.append(activity_score)
            if not activity_check.get('found', False):
                response['reason'] = "Activity title not found in certificate"
                return response
        
        # Calculate overall confidence percentage
        if confidence_factors:
            confidence_percentage = (sum(confidence_factors) / len(confidence_factors)) * 100
        else:
            confidence_percentage = verification_result['confidence_score'] * 100
        
        # Proof is valid only if confidence percentage is above threshold
        if confidence_percentage >= 50:  # 50% minimum threshold
            response['is_valid'] = True
            response['confidence_percentage'] = confidence_percentage
        else:
            response['reason'] = f"Low confidence score: {confidence_percentage:.1f}%"
        
        return response

    def is_document_proof(self, image_path: str) -> bool:
        """
        Determine if the proof is a document/certificate or an activity photo.
        Uses multiple checks to accurately distinguish between documents and activity photos.
        
        Returns:
            bool: True if it's a document (certificate, letter), False if it's an activity photo
        """
        try:
            image = Image.open(image_path)
            
            # 1. Document Structure Check
            # Check for formal document characteristics
            document_structure = self.clip_model(
                image,
                candidate_labels=[
                    "formal document with text layout and borders",
                    "certificate with official letterhead and signature",
                    "formal letter or document with structured layout",
                    "candid photo of people in action or activity"
                ]
            )
            
            # Strong indication of document
            if document_structure[0]['score'] > 0.6 and "photo" not in document_structure[0]['label'].lower():
                self.logger.info("Document detected by structure analysis")
                return True
            
            # 2. Content Type Analysis
            content_type = self.clip_model(
                image,
                candidate_labels=[
                    "certificate or award document",
                    "official letter or document",
                    "people actively participating in activity",
                    "live event or action photo",
                    "sports or physical activity photo"
                ]
            )
            
            # Clear activity photo
            if content_type[0]['score'] > 0.6 and ("activity" in content_type[0]['label'].lower() or 
                                                  "photo" in content_type[0]['label'].lower()):
                self.logger.info("Activity photo detected by content analysis")
                return False
            
            # 3. Text Density Check
            # Extract and analyze text
            extracted_text = self.extract_text_from_image(image_path)
            words = extracted_text.split()
            
            # Documents typically have more text
            if len(words) > 30:  # Threshold for minimum words in a document
                self.logger.info("Document detected by text density")
                return True
            
            # 4. Document Keywords Check
            document_keywords = {
                'certificate', 'certify', 'awarded', 'completed', 'achievement', 
                'completion', 'presented', 'hereby', 'recognition', 'award',
                'dear', 'sincerely', 'signature', 'authorized', 'date', 'issued'
            }
            
            text_lower = extracted_text.lower()
            keyword_matches = sum(1 for keyword in document_keywords if keyword in text_lower)
            
            if keyword_matches >= 2:  # At least 2 document keywords found
                self.logger.info("Document detected by keyword analysis")
                return True
            
            # 5. Activity Detection
            # Check for typical activity indicators
            activity_check = self.clip_model(
                image,
                candidate_labels=[
                    "people playing sports",
                    "workshop participants in action",
                    "active learning session",
                    "group activity or event",
                    "static document or certificate"
                ]
            )
            
            # Clear activity indicators
            if activity_check[0]['score'] > 0.6 and activity_check[0]['label'] != "static document or certificate":
                self.logger.info("Activity photo detected by activity analysis")
                return False
            
            # 6. Person Detection for Activities
            # Use object detection to check for people in action
            detections = self.object_detector(image)
            person_detections = [d for d in detections if d['label'] == "person" and d['score'] > 0.5]
            
            # Multiple people in action usually indicates an activity photo
            if len(person_detections) >= 2:
                self.logger.info("Activity photo detected by person detection")
                return False
            
            # 7. Category-based Default
            # Default handling based on category
            if hasattr(self, '_current_category'):
                # These categories typically use documents
                if self._current_category in ['courses', 'workshops', 'internship_cib', 'real_internship']:
                    self.logger.info(f"Defaulting to document based on category: {self._current_category}")
                    return True
                # These categories typically use activity photos
                elif self._current_category in ['sports', 'social', 'arts']:
                    self.logger.info(f"Defaulting to activity photo based on category: {self._current_category}")
                    return False
            
            # Default case: if we're unsure, base it on text presence
            has_significant_text = len(words) > 15
            self.logger.info(f"Defaulting based on text presence: {'document' if has_significant_text else 'activity photo'}")
            return has_significant_text
            
        except Exception as e:
            self.logger.error(f"Error in document detection: {str(e)}")
            # Default to document for document-heavy categories
            if hasattr(self, '_current_category') and self._current_category in ['courses', 'workshops', 'internship_cib', 'real_internship']:
                return True
            return False

    def verify_activity_photo(self, image_path: str, category: str, activity_title: str) -> Dict:
        """
        Verify an activity photo without text verification.
        Focus on detecting the activity and people in the image.
        """
        verification_result = {
            'verification_status': 'unverified',
            'confidence_score': 0.0,
            'details': {},
            'valid_for_time': False,
            'recommendations': []
        }

        try:
            # Define activity-specific labels based on category
            activity_labels = {
                'sports': [
                    "people playing sports",
                    "sports field or court",
                    "athletic activity",
                    "sports equipment",
                    "team sports",
                    "competitive sports",
                    "physical exercise",
                    "sports training"
                ],
                'workshops': [
                    "training class",
                    "learning activity",
                    "group discussion",
                    "presentation",
                    "workshop session",
                    "classroom learning",
                    "hands-on training",
                    "group learning activity",
                    "educational workshop",
                    "craft workshop",
                    "practical training",
                    "workshop materials"
                ],
                'social': [
                    "social gathering",
                    "group activity",
                    "community event",
                    "social event",
                    "group interaction"
                ]
            }

            # Get relevant labels for the category
            relevant_labels = activity_labels.get(category, [])

            # Add activity-specific labels based on the title
            if activity_title:
                if category == 'sports':
                    relevant_labels.extend([
                        f"playing {activity_title.lower()}",
                        f"{activity_title.lower()} game",
                        f"{activity_title.lower()} match",
                        f"{activity_title.lower()} training"
                    ])
                elif category == 'workshops':
                    relevant_labels.extend([
                        f"{activity_title.lower()} workshop",
                        f"{activity_title.lower()} training",
                        f"{activity_title.lower()} class",
                        "educational activity"
                    ])

            # Classify image content with activity-specific labels
            image = Image.open(image_path)
            activity_results = self.clip_model(image, candidate_labels=relevant_labels)
            
            # Detect objects in the image
            detections = self.object_detector(image)
            person_detections = [d for d in detections if d['label'] == "person" and d['score'] > 0.5]
            
            # Additional activity-specific object detection
            detected_objects = [d['label'].lower() for d in detections if d['score'] > 0.5]
            
            # Category-specific validation
            if category == 'sports':
                # For sports, verify presence of sports-related objects and physical activity
                sports_objects = ['ball', 'net', 'court', 'field', 'goal', 'racket', 'bat']
                has_sports_equipment = any(obj in detected_objects for obj in sports_objects)
                
                # Check if the image shows physical activity
                physical_activity = self.clip_model(
                    image,
                    candidate_labels=[
                        "physical activity or exercise",
                        "sports game in progress",
                        "athletic movement",
                        "static or seated activity"
                    ]
                )
                
                is_physical = physical_activity[0]['label'] != "static or seated activity" and physical_activity[0]['score'] > 0.6
                
                if not (has_sports_equipment or is_physical):
                    verification_result['recommendations'].append(
                        "The image should show active sports participation or sports equipment"
                    )
                    return verification_result
                    
            elif category == 'workshops':
                # For workshops, verify educational/training setting
                workshop_objects = ['table', 'desk', 'chair', 'board', 'screen', 'computer', 'tool']
                has_workshop_equipment = any(obj in detected_objects for obj in workshop_objects)
                
                # Check if the image shows learning/workshop activity
                workshop_activity = self.clip_model(
                    image,
                    candidate_labels=[
                        "educational or training activity",
                        "workshop or class in progress",
                        "hands-on learning",
                        "unrelated activity"
                    ]
                )
                
                is_workshop = workshop_activity[0]['label'] != "unrelated activity" and workshop_activity[0]['score'] > 0.6
                
                if not (has_workshop_equipment or is_workshop):
                    verification_result['recommendations'].append(
                        "The image should show workshop participation or training materials"
                    )
                    return verification_result

            # Calculate confidence based on multiple factors
            confidence_factors = []
            
            # 1. Activity match confidence
            activity_confidence = activity_results[0]['score']
            confidence_factors.append(activity_confidence)
            
            # 2. Person presence confidence
            person_confidence = min(len(person_detections) / 2, 1.0) if person_detections else 0.0
            confidence_factors.append(person_confidence)
            
            # 3. Category-specific confidence
            if category == 'sports':
                sports_confidence = 1.0 if (has_sports_equipment or is_physical) else 0.3
                confidence_factors.append(sports_confidence)
            elif category == 'workshops':
                workshop_confidence = 1.0 if (has_workshop_equipment or is_workshop) else 0.3
                confidence_factors.append(workshop_confidence)
            
            # Calculate final confidence score
            verification_result['confidence_score'] = sum(confidence_factors) / len(confidence_factors)
            
            # Store detection details
            verification_result['details'].update({
                'activity_match': {
                    'detected_activity': activity_results[0]['label'],
                    'confidence': activity_confidence
                },
                'person_detection': {
                    'people_count': len(person_detections),
                    'confidence': person_confidence
                },
                'detected_objects': detected_objects
            })

            # Determine verification status
            if verification_result['confidence_score'] >= 0.6:
                verification_result['verification_status'] = 'verified'
                verification_result['valid_for_time'] = True
                verification_result['time_validation'] = {
                    'is_valid': True,
                    'confidence_percentage': verification_result['confidence_score'] * 100
                }
            else:
                if not person_detections:
                    verification_result['recommendations'].append(
                        "The image should clearly show people participating in the activity"
                    )
                if activity_confidence < 0.5:
                    verification_result['recommendations'].append(
                        f"The activity in the image should clearly show {activity_title or category}"
                    )

            return verification_result

        except Exception as e:
            self.logger.error(f"Error verifying activity photo: {str(e)}")
            verification_result['verification_status'] = 'error'
            verification_result['error'] = str(e)
            return verification_result

    def verify_proof(self, image_path: str, category: str, activity_title: str = "", user_name: str = "") -> Dict:
        """
        Verify a proof/certificate.
        
        Args:
            image_path (str): Path to the image file
            category (str): Category of the activity
            activity_title (str, optional): Title of the activity
            user_name (str, optional): Name of the user to verify
            
        Returns:
            Dict: Verification result containing:
                - verification_status: str ('verified', 'partially_verified', or 'unverified')
                - confidence_score: float (0-1)
                - details: Dict with analysis details
                - recommendations: List[str] with improvement suggestions
        """
        try:
            self.logger.info(f"Starting verification for category: {category}, "
                           f"user: '{user_name}', activity: '{activity_title}'")
            
            # Extract text from image
            extracted_text = self.extract_text_from_image(image_path)
            text_length = len(extracted_text)
            self.logger.info(f"Extracted text length: {text_length} characters")
            self.logger.info(f"Full extracted text: {extracted_text}")
            
            # Initialize result dictionary
            result = {
                'verification_status': 'unverified',
                'confidence_score': 0.0,
                'details': {},
                'recommendations': [],
                'category': category,
                'activity_title': activity_title,
                'user_name': user_name
            }
            
            # Analyze based on category
            if category in ['courses', 'workshops']:
                # For courses and workshops, expect a certificate
                if self.is_document_proof(image_path):
                    self.logger.info("Document detected by structure analysis")
                    result = self._verify_certificate(
                        extracted_text, 
                        category, 
                        activity_title, 
                        user_name,
                        result
                    )
                else:
                    result['recommendations'].append(
                        "Please upload a clear image of the certificate"
                    )
            else:
                # For other categories, analyze activity photo
                if self.is_document_proof(image_path):
                    self.logger.info("Activity photo detected by activity analysis")
                    result = self._verify_activity_photo(
                        image_path,
                        extracted_text,
                        category,
                        activity_title,
                        result
                    )
                else:
                    result['recommendations'].append(
                        "Please upload a clear photo of the activity"
                    )
            
            # Save result to database
            try:
                self.logger.info("Attempting to save verification result to database at verification_data.db")
                self.logger.info(f"Verification data: status={result['verification_status']}, "
                               f"category={category}, user={user_name}")
                
                verification_id = self.verification_service.save_verification_result(result)
                result['id'] = verification_id
                self.logger.info(f"Successfully saved verification result as submission with ID: {verification_id}")
                
            except Exception as e:
                self.logger.error(f"Error saving to database: {str(e)}")
                # Continue even if database save fails
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error during verification: {str(e)}")
            return {
                'verification_status': 'error',
                'error': str(e),
                'category': category,
                'user_name': user_name
            }

    def get_verification_by_id(self, verification_id: int) -> Optional[Dict]:
        """Get verification result by ID"""
        return self.verification_service.get_verification_by_id(verification_id)
        
    def get_verifications_by_user(self, user_name: str) -> List[Dict]:
        """Get all verifications for a user"""
        return self.verification_service.get_verifications_by_user(user_name)
        
    def get_verifications_by_category(self, category: str) -> List[Dict]:
        """Get all verifications for a category"""
        return self.verification_service.get_verifications_by_category(category)
        
    def get_verification_stats(self) -> Dict:
        """Get verification statistics"""
        return self.verification_service.get_verification_stats()
        
    def get_recent_verifications(self, limit: int = 10) -> List[Dict]:
        """Get most recent verifications"""
        return self.verification_service.get_recent_verifications(limit)
        
    def search_verifications(self, **kwargs) -> List[Dict]:
        """Search verifications with filters"""
        return self.verification_service.search_verifications(**kwargs)
        
    def delete_verification(self, verification_id: int) -> bool:
        """Delete a verification record"""
        return self.verification_service.delete_verification(verification_id)
        
    def cleanup_old_verifications(self, days_to_keep: int = 30) -> int:
        """Clean up old verification records"""
        return self.verification_service.cleanup_old_verifications(days_to_keep)
        
    def export_verifications_to_json(self, output_file: str) -> None:
        """Export verifications to JSON file"""
        self.verification_service.export_verifications_to_json(output_file)

    def calculate_confidence_score(self, text_relevance: float, activity_score: float, name_score: float) -> float:
        """
        Calculate overall confidence score based on multiple factors.
        
        Args:
            text_relevance (float): Score from text keyword analysis (0-1)
            activity_score (float): Score from activity title matching (0-1)
            name_score (float): Score from name matching (0-1)
            
        Returns:
            float: Overall confidence score (0-1)
        """
        # Weight factors (total should be 1.0)
        TEXT_WEIGHT = 0.3      # Text relevance weight (30%)
        ACTIVITY_WEIGHT = 0.35  # Activity title match weight (35%)
        NAME_WEIGHT = 0.35     # Name match weight (35%)
        
        # Calculate weighted score
        weighted_score = (
            text_relevance * TEXT_WEIGHT +
            activity_score * ACTIVITY_WEIGHT +
            name_score * NAME_WEIGHT
        )
        
        return min(weighted_score, 1.0)  # Cap at 1.0

    def _get_validation_reason(self, name_verified: bool, activity_verified: bool, confidence_score: float) -> str:
        """Get reason for time validation result"""
        if not name_verified:
            return "Name not found in certificate"
        if not activity_verified:
            return "Activity title not found in certificate"
        if confidence_score < 0.7:
            return f"Low confidence score: {confidence_score * 100:.1f}%"
        return "Valid proof"

    def generate_recommendations(self, text_verification: Dict, activity_check: Optional[Dict], name_check: Optional[Dict]) -> List[str]:
        """Generate recommendations for improving verification results"""
        recommendations = []
        
        # Text verification recommendations
        if text_verification['relevance_score'] < 0.5:
            recommendations.append(
                "The document should contain more relevant keywords for the selected category"
            )
        
        # Activity title recommendations
        if activity_check and not activity_check['found']:
            recommendations.append(
                "The activity title should be clearly visible in the document"
            )
        elif activity_check and activity_check['score'] < 0.5:
            recommendations.append(
                "The activity title in the document should match more closely with the provided title"
            )
        
        # Name recommendations
        if name_check and not name_check['found']:
            recommendations.append(
                "The name should be clearly visible in the document"
            )
        elif name_check and name_check['score'] < 0.5:
            recommendations.append(
                "The name in the document should match more closely with the provided name"
            )
        
        return recommendations


