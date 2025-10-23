"""
Simple interface for human review of flagged content
"""

import json
from datetime import datetime
from typing import List, Dict, Any

class HumanReviewInterface:
    def __init__(self, review_file: str = "human_reviews.json"):
        self.pending_reviews = []
        self.review_decisions = {}
        self.review_file = review_file
        self.load_existing_reviews()
    
    def load_existing_reviews(self):
        """Load existing review decisions from file"""
        try:
            with open(self.review_file, 'r') as f:
                self.review_decisions = json.load(f)
        except FileNotFoundError:
            self.review_decisions = {}
    
    def save_reviews(self):
        """Save review decisions to file"""
        with open(self.review_file, 'w') as f:
            json.dump(self.review_decisions, f, indent=2)
    
    def add_for_review(self, analysis_result: Dict[str, Any]):
        """Add content that needs human review"""
        if analysis_result['category'] == 'needs_review':
            review_item = {
                'id': len(self.pending_reviews) + 1,
                'timestamp': datetime.now().isoformat(),
                'analysis': analysis_result
            }
            self.pending_reviews.append(review_item)
    
    def display_pending_reviews(self):
        """Display all content needing human review"""
        if not self.pending_reviews:
            print("No pending reviews! 🎉")
            return
        
        print("\n" + "="*60)
        print("CONTENT NEEDING HUMAN REVIEW")
        print("="*60)
        
        for review in self.pending_reviews:
            self.display_single_review(review)
            
        self.save_reviews()
    
    def display_single_review(self, review: Dict):
        """Display a single review item and get decision"""
        analysis = review['analysis']
        
        print(f"\n📝 REVIEW ITEM {review['id']}/{len(self.pending_reviews)}")
        print(f"Text: {analysis['text']}")
        print(f"Auto-Category: {analysis['category']}")
        print(f"Confidence: {analysis['confidence']:.2f}")
        print(f"Verification Score: {analysis['verification_score']:.2f}")
        print(f"Coherence Score: {analysis['coherence_score']:.2f}")
        
        # Show conceptual breakdown
        if 'conceptual_breakdown' in analysis:
            breakdown = analysis['conceptual_breakdown']
            print("\nConceptual Breakdown:")
            print(f"  Agent: {breakdown.get('primary_agent', 'None')}")
            print(f"  Action: {breakdown.get('primary_action', 'None')}")
            print(f"  Objects: {', '.join(breakdown.get('key_objects', []))}")
            if breakdown.get('conceptual_gaps'):
                print(f"  Gaps: {', '.join(breakdown['conceptual_gaps'])}")
        
        print("\nOptions:")
        print("  [f] fact     [s] speculation     [r] false     [k] skip")
        
        while True:
            decision = input("\nYour decision: ").strip().lower()
            if decision in ['f', 'fact']:
                final_decision = 'fact'
                break
            elif decision in ['s', 'speculation']:
                final_decision = 'speculation'
                break
            elif decision in ['r', 'false']:
                final_decision = 'false'
                break
            elif decision in ['k', 'skip']:
                return
            else:
                print("Invalid option. Please choose f, s, r, or k.")
        
        # Store decision
        self.review_decisions[analysis['text']] = {
            'decision': final_decision,
            'reviewed_at': datetime.now().isoformat(),
            'review_id': review['id']
        }
        
        print(f"✓ Marked as: {final_decision}")
    
    def get_review_stats(self) -> Dict[str, Any]:
        """Get statistics about review decisions"""
        stats = {
            'total_reviewed': len(self.review_decisions),
            'decisions': {},
            'recent_activity': []
        }
        
        for text, decision_data in self.review_decisions.items():
            decision = decision_data['decision']
            stats['decisions'][decision] = stats['decisions'].get(decision, 0) + 1
        
        return stats
    
    def get_final_dataset(self, processed_data: List[Dict]) -> List[Dict]:
        """Apply human decisions to create final training dataset"""
        final_data = []
        
        for item in processed_data:
            text = item['text']
            if text in self.review_decisions:
                item['final_category'] = self.review_decisions[text]['decision']
                item['human_reviewed'] = True
            else:
                item['final_category'] = item['category']
                item['human_reviewed'] = False
                
            # Only include fact and speculation in training data
            if item['final_category'] in ['fact', 'speculation']:
                final_data.append(item)
        
        return final_data
