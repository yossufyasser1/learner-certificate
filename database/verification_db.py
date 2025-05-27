import sqlite3
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

class VerificationDatabase:
    """Database manager for verification results"""
    
    def __init__(self, db_path: str = "verification_data.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.initialize_database()
        
    def initialize_database(self):
        """Initialize SQLite database with necessary tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create users table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_name TEXT NOT NULL UNIQUE,
                        total_hours REAL DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create submissions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS submissions (
                        submission_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        category TEXT NOT NULL,
                        activity_title TEXT,
                        verification_status TEXT DEFAULT 'pending',
                        original_time INTEGER,  -- Time in minutes
                        calculated_time INTEGER,  -- Time in minutes after verification
                        verification_report TEXT,
                        confidence_score REAL,
                        recommendations TEXT,
                        image_path TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users(user_id)
                    )
                """)
                
                # Create trigger to update user's total_hours
                cursor.execute("""
                    CREATE TRIGGER IF NOT EXISTS update_user_hours
                    AFTER INSERT ON submissions
                    BEGIN
                        UPDATE users 
                        SET 
                            total_hours = (
                                SELECT ROUND(CAST(SUM(calculated_time) AS FLOAT) / 60, 2)
                                FROM submissions 
                                WHERE user_id = NEW.user_id 
                                AND verification_status = 'verified'
                            ),
                            updated_at = CURRENT_TIMESTAMP
                        WHERE user_id = NEW.user_id;
                    END;
                """)
                
                conn.commit()
                self.logger.info("Database initialized successfully")
                
        except sqlite3.Error as e:
            self.logger.error(f"Database initialization error: {str(e)}")
            raise
            
    def create_user(self, user_name: str) -> int:
        """Create a new user"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO users (user_name) VALUES (?)",
                    (user_name,)
                )
                conn.commit()
                return cursor.lastrowid
        except sqlite3.IntegrityError:
            # User already exists, get their ID
            cursor = conn.cursor()
            cursor.execute("SELECT user_id FROM users WHERE user_name = ?", (user_name,))
            return cursor.fetchone()[0]
            
    def save_submission(self, submission_data: Dict) -> int:
        """Save a new submission"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Ensure user exists
                user_id = self.create_user(submission_data['user_name'])
                
                cursor.execute("""
                    INSERT INTO submissions (
                        user_id, category, activity_title, verification_status,
                        original_time, calculated_time, verification_report,
                        confidence_score, recommendations, image_path
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user_id,
                    submission_data['category'],
                    submission_data.get('activity_title', ''),
                    submission_data.get('verification_status', 'pending'),
                    submission_data.get('original_time', 0),
                    submission_data.get('calculated_time', 0),
                    json.dumps(submission_data.get('verification_report', {})),
                    submission_data.get('confidence_score', 0.0),
                    json.dumps(submission_data.get('recommendations', [])),
                    submission_data.get('image_path', '')
                ))
                
                submission_id = cursor.lastrowid
                conn.commit()
                return submission_id
                
        except sqlite3.Error as e:
            self.logger.error(f"Error saving submission: {str(e)}")
            raise
            
    def get_user_submissions(self, user_name: str) -> List[Dict]:
        """Get all submissions for a user"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT s.*, u.user_name, u.total_hours
                    FROM submissions s
                    JOIN users u ON s.user_id = u.user_id
                    WHERE u.user_name = ?
                    ORDER BY s.created_at DESC
                """, (user_name,))
                
                results = []
                for row in cursor.fetchall():
                    result = dict(row)
                    result['verification_report'] = json.loads(result['verification_report']) if result['verification_report'] else {}
                    result['recommendations'] = json.loads(result['recommendations']) if result['recommendations'] else []
                    results.append(result)
                
                return results
                
        except sqlite3.Error as e:
            self.logger.error(f"Error getting user submissions: {str(e)}")
            raise
            
    def get_user_stats(self, user_name: str) -> Dict:
        """Get user statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        u.total_hours,
                        COUNT(s.submission_id) as total_submissions,
                        SUM(CASE WHEN s.verification_status = 'verified' THEN 1 ELSE 0 END) as verified_submissions,
                        AVG(s.confidence_score) as avg_confidence
                    FROM users u
                    LEFT JOIN submissions s ON u.user_id = s.user_id
                    WHERE u.user_name = ?
                    GROUP BY u.user_id
                """, (user_name,))
                
                row = cursor.fetchone()
                if not row:
                    return {
                        'total_hours': 0,
                        'total_submissions': 0,
                        'verified_submissions': 0,
                        'avg_confidence': 0
                    }
                    
                return {
                    'total_hours': row[0] or 0,
                    'total_submissions': row[1] or 0,
                    'verified_submissions': row[2] or 0,
                    'avg_confidence': row[3] or 0
                }
                
        except sqlite3.Error as e:
            self.logger.error(f"Error getting user stats: {str(e)}")
            raise
            
    def save_verification_result(self, result: Dict) -> int:
        """Save verification result to database"""
        try:
            self.logger.info(f"Attempting to save verification result to database at {self.db_path}")
            self.logger.info(f"Verification data: status={result.get('verification_status')}, "
                           f"category={result.get('category')}, user={result.get('user_name')}")
            
            # Convert verification status to is_verified boolean
            is_verified = result['verification_status'] == 'verified'
            
            # Prepare submission data
            submission_data = {
                'user_name': result.get('user_name', ''),
                'category': result['category'],
                'activity_title': result.get('activity_title', ''),
                'verification_status': result['verification_status'],
                'original_time': 0,  # This would need to be passed in if needed
                'calculated_time': 0,  # This would need to be passed in if needed
                'verification_report': result.get('verification_report', {}),
                'confidence_score': result.get('confidence_score', 0.0),
                'recommendations': result.get('recommendations', []),
                'image_path': result.get('image_path', '')
            }
            
            # Use the existing save_submission method
            submission_id = self.save_submission(submission_data)
            self.logger.info(f"Successfully saved verification result as submission with ID: {submission_id}")
            return submission_id
                
        except sqlite3.Error as e:
            self.logger.error(f"SQLite error saving verification result: {str(e)}")
            self.logger.error(f"Database path: {self.db_path}")
            self.logger.error(f"Result data: {result}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error saving verification result: {str(e)}")
            self.logger.error(f"Database path: {self.db_path}")
            self.logger.error(f"Result data: {result}")
            raise
            
    def get_verification_by_id(self, verification_id: int) -> Optional[Dict]:
        """Retrieve verification result by ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT s.*, u.user_name
                    FROM submissions s
                    JOIN users u ON s.user_id = u.user_id
                    WHERE s.submission_id = ?
                """, (verification_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                result = dict(row)
                result['verification_report'] = json.loads(result['verification_report']) if result['verification_report'] else {}
                result['recommendations'] = json.loads(result['recommendations']) if result['recommendations'] else []
                
                return result
                
        except sqlite3.Error as e:
            self.logger.error(f"Error retrieving verification result: {str(e)}")
            raise
            
    def get_verifications_by_user(self, user_name: str) -> List[Dict]:
        """Retrieve all verification results for a specific user"""
        return self.get_user_submissions(user_name)
            
    def get_verifications_by_category(self, category: str) -> List[Dict]:
        """Retrieve all verification results for a specific category"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT s.*, u.user_name
                    FROM submissions s
                    JOIN users u ON s.user_id = u.user_id
                    WHERE s.category = ?
                    ORDER BY s.created_at DESC
                """, (category,))
                
                results = []
                for row in cursor.fetchall():
                    result = dict(row)
                    result['verification_report'] = json.loads(result['verification_report']) if result['verification_report'] else {}
                    result['recommendations'] = json.loads(result['recommendations']) if result['recommendations'] else []
                    results.append(result)
                
                return results
                
        except sqlite3.Error as e:
            self.logger.error(f"Error retrieving category verifications: {str(e)}")
            raise
            
    def get_verification_stats(self) -> Dict:
        """Get statistics about verifications in the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get total counts
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN verification_status = 'verified' THEN 1 ELSE 0 END) as verified,
                        AVG(confidence_score) as avg_confidence
                    FROM submissions
                """)
                
                row = cursor.fetchone()
                
                # Get category breakdown
                cursor.execute("""
                    SELECT category, COUNT(*) as count
                    FROM submissions
                    GROUP BY category
                    ORDER BY count DESC
                """)
                
                category_stats = {row[0]: row[1] for row in cursor.fetchall()}
                
                return {
                    'total_verifications': row[0] or 0,
                    'verified_count': row[1] or 0,
                    'average_confidence': float(row[2]) if row[2] is not None else 0.0,
                    'category_breakdown': category_stats
                }
                
        except sqlite3.Error as e:
            self.logger.error(f"Error getting verification stats: {str(e)}")
            raise
            
    def get_recent_verifications(self, limit: int = 10) -> List[Dict]:
        """Get most recent verifications"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT s.*, u.user_name
                    FROM submissions s
                    JOIN users u ON s.user_id = u.user_id
                    ORDER BY s.created_at DESC
                    LIMIT ?
                """, (limit,))
                
                results = []
                for row in cursor.fetchall():
                    result = dict(row)
                    result['verification_report'] = json.loads(result['verification_report']) if result['verification_report'] else {}
                    result['recommendations'] = json.loads(result['recommendations']) if result['recommendations'] else []
                    results.append(result)
                
                return results
                
        except sqlite3.Error as e:
            self.logger.error(f"Error getting recent verifications: {str(e)}")
            raise
            
    def search_verifications(self, 
                           user_name: Optional[str] = None,
                           category: Optional[str] = None,
                           status: Optional[str] = None,
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> List[Dict]:
        """Search verifications with various filters"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                query = """
                    SELECT s.*, u.user_name
                    FROM submissions s
                    JOIN users u ON s.user_id = u.user_id
                    WHERE 1=1
                """
                params = []
                
                if user_name:
                    query += " AND u.user_name LIKE ?"
                    params.append(f"%{user_name}%")
                
                if category:
                    query += " AND s.category = ?"
                    params.append(category)
                
                if status:
                    is_verified = status.lower() == 'verified'
                    query += " AND s.verification_status = ?"
                    params.append(status)
                
                if start_date:
                    query += " AND s.created_at >= ?"
                    params.append(start_date)
                
                if end_date:
                    query += " AND s.created_at <= ?"
                    params.append(end_date)
                
                query += " ORDER BY s.created_at DESC"
                
                cursor.execute(query, params)
                
                results = []
                for row in cursor.fetchall():
                    result = dict(row)
                    result['verification_report'] = json.loads(result['verification_report']) if result['verification_report'] else {}
                    result['recommendations'] = json.loads(result['recommendations']) if result['recommendations'] else []
                    results.append(result)
                
                return results
                
        except sqlite3.Error as e:
            self.logger.error(f"Error searching verifications: {str(e)}")
            raise
            
    def delete_verification(self, verification_id: int) -> bool:
        """Delete a verification record"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("DELETE FROM submissions WHERE submission_id = ?", 
                             (verification_id,))
                
                deleted = cursor.rowcount > 0
                conn.commit()
                
                if deleted:
                    self.logger.info(f"Verification {verification_id} deleted successfully")
                else:
                    self.logger.warning(f"Verification {verification_id} not found")
                
                return deleted
                
        except sqlite3.Error as e:
            self.logger.error(f"Error deleting verification: {str(e)}")
            raise
            
    def cleanup_old_verifications(self, days_to_keep: int = 30) -> int:
        """Clean up old verification records"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    DELETE FROM submissions 
                    WHERE datetime(created_at) < datetime('now', ?)
                """, (f'-{days_to_keep} days',))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                self.logger.info(f"Cleaned up {deleted_count} old verification records")
                return deleted_count
                
        except sqlite3.Error as e:
            self.logger.error(f"Error cleaning up old verifications: {str(e)}")
            raise
            
    def export_verifications_to_json(self, output_file: str) -> None:
        """Export all verifications to a JSON file"""
        try:
            verifications = self.get_recent_verifications(limit=1000000)  # Get all verifications
            
            with open(output_file, 'w') as f:
                json.dump({
                    'verifications': verifications,
                    'stats': self.get_verification_stats(),
                    'exported_at': datetime.now().isoformat()
                }, f, indent=2)
                
            self.logger.info(f"Exported {len(verifications)} verifications to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error exporting verifications: {str(e)}")
            raise

    def update_verification_status(self, verification_id: int, status: str) -> None:
        """
        Update the status of a verification record.
        
        Args:
            verification_id (int): ID of the verification to update
            status (str): New status to set
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE verifications 
                    SET verification_status = ?, 
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (status, verification_id)
                )
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error updating verification status: {str(e)}")
            raise 