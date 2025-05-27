from proof_verifier import ProofVerifier
import logging

def main():
    # Initialize the verifier (this will automatically initialize the database)
    verifier = ProofVerifier(db_path="verification_data.db")
    
    # Example: Verify a certificate
    result = verifier.verify_proof(
        image_path="path/to/your/certificate.jpg",
        category="courses",
        activity_title="Python Programming",
        user_name="John Doe"
    )
    
    # Print verification result
    print("\nVerification Result:")
    print(f"Status: {result['verification_status']}")
    print(f"Confidence Score: {result['confidence_score']:.2f}")
    print(f"Valid for Time: {result['valid_for_time']}")
    
    # Example: Get verification statistics
    stats = verifier.get_verification_stats()
    print("\nVerification Statistics:")
    print(f"Total Verifications: {stats['total_verifications']}")
    print(f"Verified Count: {stats['verified_count']}")
    print(f"Average Confidence: {stats['average_confidence']:.2f}")
    
    # Example: Search verifications
    print("\nRecent Verifications:")
    recent = verifier.get_recent_verifications(limit=5)
    for v in recent:
        print(f"- {v['timestamp']}: {v['verification_status']} ({v['category']})")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    main() 