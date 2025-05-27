from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from proof_verifier import ProofVerifier
from database import VerificationService
import tempfile
import os
from typing import Optional, List, Dict
import logging
import hashlib
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CertificateVerificationAPI:
    def __init__(self):
        logger.info("Initializing CertificateVerificationAPI")
        logger.info("Creating ProofVerifier instance and VerificationService")
        self.verifier = ProofVerifier()
        self.verification_service = VerificationService()
        logger.info("Services created successfully")
        self.certificate_cache = {}
        self.app = self._create_app()
        logger.info("CertificateVerificationAPI initialization complete")
    
    def _create_app(self) -> FastAPI:
        """Create and configure FastAPI application"""
        app = FastAPI(
            title="Certificate Verification API",
            description="API for verifying certificates and activity proofs",
            version="1.0.0"
        )
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Register routes
        app.post("/verify/")(self.verify_proof)
        app.get("/categories/")(self.get_categories)
        app.get("/health")(self.health_check)
        app.get("/user/{user_name}/submissions")(self.get_user_submissions)
        app.get("/user/{user_name}/stats")(self.get_user_stats)
        app.post("/submit_all/")(self.submit_all_activities)
        
        return app
    
    def _get_file_hash(self, content: bytes) -> str:
        """Generate hash for file content"""
        return hashlib.sha256(content).hexdigest()
    
    def _get_cache_key(self, file_hash: str, category: str, activity_title: str, user_name: str) -> str:
        """Generate cache key for verification request"""
        return f"{file_hash}:{category}:{activity_title}:{user_name}"
    
    async def verify_proof(
        self,
        file: UploadFile = File(...),
        category: str = Form(...),
        activity_title: Optional[str] = Form(None),
        user_name: Optional[str] = Form(None)
    ):
        """Verify a proof/certificate using the ProofVerifier"""
        try:
            logger.info(f"Verification request - Category: {category}, "
                       f"User: '{user_name}', Activity: '{activity_title}'")
            
            # Validate inputs
            if not category:
                raise HTTPException(status_code=400, detail="Category is required")
            
            # Read and hash file content
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="Empty file uploaded")
            
            file_hash = self._get_file_hash(content)
            cache_key = self._get_cache_key(file_hash, category, activity_title, user_name)
            
            # Check cache
            if cache_key in self.certificate_cache:
                logger.info(f"Using cached result for {cache_key}")
                return self.certificate_cache[cache_key]
            
            # Save file temporarily
            suffix = os.path.splitext(file.filename or "")[1] if file.filename else ".tmp"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(content)
                temp_path = temp_file.name
            
            logger.info(f"Saved uploaded file to: {temp_path}")
            
            # Verify the proof
            result = self.verifier.verify_proof(
                image_path=temp_path,
                category=category,
                activity_title=activity_title or "",
                user_name=user_name or ""
            )
            
            # Ensure result has verification_status
            if 'verification_status' not in result:
                result['verification_status'] = 'verified' if result.get('is_verified', False) else 'unverified'
            
            # Ensure result is saved to database (in case it wasn't saved in verify_proof)
            if 'id' not in result:
                verification_id = self.verification_service.save_verification_result(result)
                result['id'] = verification_id
                logger.info(f"Explicitly saved verification result with ID: {verification_id}")
            
            # Cache the result
            self.certificate_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing verification request: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        
        finally:
            # Clean up temporary file
            try:
                if 'temp_path' in locals():
                    os.unlink(temp_path)
            except Exception as e:
                logger.error(f"Error cleaning up temporary file: {str(e)}")
    
    async def get_categories(self):
        """Get available verification categories"""
        return {
            "categories": [
                {"id": "courses", "name": "Course", "icon": "📚"},
                {"id": "workshops", "name": "Workshop", "icon": "🔧"},
                {"id": "student_activities", "name": "Student Activity", "icon": "🎓"},
                {"id": "volunteering", "name": "Volunteering", "icon": "🤝"},
                {"id": "internships", "name": "Internship", "icon": "💼"},
                {"id": "sports", "name": "Sports", "icon": "⚽"},
                {"id": "arts", "name": "Arts", "icon": "🎨"},
                {"id": "others", "name": "Others", "icon": "🔄"}
            ]
        }
    
    async def health_check(self):
        """API health check endpoint"""
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    
    async def get_user_submissions(self, user_name: str):
        """Get all submissions for a user"""
        try:
            submissions = self.verification_service.get_verifications_by_user(user_name)
            return {"submissions": submissions}
        except Exception as e:
            logger.error(f"Error getting user submissions: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_user_stats(self, user_name: str):
        """Get statistics for a user"""
        try:
            stats = self.verification_service.get_verification_stats()
            return {"stats": stats}
        except Exception as e:
            logger.error(f"Error getting user stats: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def submit_all_activities(self, activities: List[Dict] = Body(...)):
        """
        Submit all verified activities in a batch.
        
        Expected request body format:
        [
            {
                "verification_id": int,
                "category": str,
                "activity_title": str,
                "time_spent": float,
                "time_unit": str
            },
            ...
        ]
        """
        try:
            logger.info(f"Received batch submission request for {len(activities)} activities")
            
            # Validate and process each activity
            processed_activities = []
            total_time = 0
            
            for activity in activities:
                # Get verification result from database
                verification_id = activity.get('verification_id')
                if not verification_id:
                    raise HTTPException(status_code=400, detail="Missing verification_id")
                    
                verification = self.verification_service.get_verification_by_id(verification_id)
                if not verification:
                    raise HTTPException(
                        status_code=404, 
                        detail=f"Verification not found: {verification_id}"
                    )
                
                # If verification status is not 'verified', skip it
                if verification.get('verification_status') != 'verified':
                    logger.warning(f"Activity {verification_id} not verified, skipping")
                    continue
                
                # Calculate equivalent time based on category multiplier
                category = activity.get('category', '')
                time_spent = float(activity.get('time_spent', 0))
                time_unit = activity.get('time_unit', 'hours')
                
                # Convert months to hours if needed
                if time_unit == 'months':
                    time_spent *= 30 * 24  # Convert months to hours (30 days * 24 hours)
                
                # Apply category multiplier
                multipliers = {
                    'courses': 3,
                    'workshops': 4,
                    'student_activities_aisec': 7,
                    'other_student_activities': 4,
                    'volunteering_cop': 15,
                    'volunteering_icareer': 10,
                    'youtube': 2,
                    'internship_cib': 8,
                    'real_internship': 50,
                    'time_spent': 1,
                    'sports': 2,
                    'social': 2,
                    'arts': 2,
                    'others': 1.5
                }
                
                multiplier = multipliers.get(category, 1)
                equivalent_time = time_spent * multiplier
                
                # Add to processed activities
                processed_activity = {
                    **activity,
                    'verification_status': verification['verification_status'],
                    'confidence_score': verification.get('confidence_score', 1.0),
                    'equivalent_time': equivalent_time,
                    'multiplier': multiplier
                }
                processed_activities.append(processed_activity)
                total_time += equivalent_time
            
            if not processed_activities:
                raise HTTPException(
                    status_code=400,
                    detail="No valid activities to submit"
                )
            
            # Update verification status in database
            for activity in processed_activities:
                self.verification_service.update_verification_status(
                    activity['verification_id'],
                    'submitted'
                )
            
            return {
                'status': 'success',
                'message': 'Activities submitted successfully',
                'total_activities': len(processed_activities),
                'total_equivalent_time': total_time,
                'processed_activities': processed_activities
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing batch submission: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing submission: {str(e)}"
            )

def create_app() -> FastAPI:
    """Factory function to create the FastAPI app"""
    api = CertificateVerificationAPI()
    return api.app

def find_available_port(start_port: int = 8002, max_attempts: int = 10) -> int:
    """Find an available port starting from start_port"""
    import socket
    
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    
    raise RuntimeError(f"Could not find available port in range {start_port}-{start_port + max_attempts}")

def start_server(host: str = "0.0.0.0", port: int = None):
    """Start the FastAPI server"""
    if port is None:
        port = find_available_port()
        logger.info(f"Auto-selected port: {port}")
    
    app = create_app()
    
    try:
        logger.info(f"Starting server on {host}:{port}")
        uvicorn.run(app, host=host, port=port)
    except OSError as e:
        if "address already in use" in str(e).lower() or "10048" in str(e):
            logger.error(f"Port {port} is already in use. Trying to find available port...")
            available_port = find_available_port(port + 1)
            logger.info(f"Starting server on available port: {available_port}")
            uvicorn.run(app, host=host, port=available_port)
        else:
            raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Certificate Verification API Server")
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, help='Port to bind to (auto-detect if not specified)')
    
    args = parser.parse_args()
    start_server(args.host, args.port)