from typing import Dict, List, Optional
from .verification_db import VerificationDatabase
import logging

logger = logging.getLogger(__name__)

class VerificationService:
    """Service layer for handling verification data persistence"""
    
    def __init__(self, db_path: str = "verification_data.db"):
        self.db = VerificationDatabase(db_path)
    
    def save_verification_result(self, result: Dict) -> int:
        """Save verification result and return the ID"""
        return self.db.save_verification_result(result)
    
    def get_verification_by_id(self, verification_id: int) -> Optional[Dict]:
        """Get verification result by ID"""
        return self.db.get_verification_by_id(verification_id)
    
    def get_verifications_by_user(self, user_name: str) -> List[Dict]:
        """Get all verifications for a user"""
        return self.db.get_verifications_by_user(user_name)
    
    def get_verifications_by_category(self, category: str) -> List[Dict]:
        """Get all verifications for a category"""
        return self.db.get_verifications_by_category(category)
    
    def get_verification_stats(self) -> Dict:
        """Get verification statistics"""
        return self.db.get_verification_stats()
    
    def get_recent_verifications(self, limit: int = 10) -> List[Dict]:
        """Get most recent verifications"""
        return self.db.get_recent_verifications(limit)
    
    def search_verifications(self, **kwargs) -> List[Dict]:
        """Search verifications with filters"""
        return self.db.search_verifications(**kwargs)
    
    def delete_verification(self, verification_id: int) -> bool:
        """Delete a verification record"""
        return self.db.delete_verification(verification_id)
    
    def cleanup_old_verifications(self, days_to_keep: int = 30) -> int:
        """Clean up old verification records"""
        return self.db.cleanup_old_verifications(days_to_keep)
    
    def export_verifications_to_json(self, output_file: str) -> None:
        """Export verifications to JSON file"""
        self.db.export_verifications_to_json(output_file)
    
    def update_verification_status(self, verification_id: int, status: str) -> bool:
        """
        Update the status of a verification record.
        
        Args:
            verification_id (int): ID of the verification to update
            status (str): New status to set
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        try:
            self.db.update_verification_status(verification_id, status)
            return True
        except Exception as e:
            logger.error(f"Error updating verification status: {str(e)}")
            return False 