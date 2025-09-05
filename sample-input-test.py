# Quick Test Script - Test Your AI Agent
# Use this to test different input scenarios

def test_scheduling_scenarios():
    """Test different ways users might interact with the scheduling agent"""
    
    print("🏥 AI Scheduling Agent - Input Testing Guide")
    print("=" * 60)
    
    test_cases = [
        {
            "scenario": "📅 SCHEDULING - Natural Language",
            "inputs": [
                "Hi, I am John Doe, I would like to book an appointment",
                "03/15/1985", 
                "123 Main Street, New York, NY 10001",
                "john.doe@gmail.com",
                "Dr. Emily Chen",
                "1",
                "Blue Cross Blue Shield, member ID 123456789, group 987654"
            ]
        },
        {
            "scenario": "📅 SCHEDULING - Step by Step", 
            "inputs": [
                "Hello, I need to schedule an appointment",
                "Sarah",
                "Smith", 
                "07/22/1990",
                "456 Oak Avenue, Los Angeles, CA 90210",
                "sarah.smith@gmail.com", 
                "Dr. David Rodriguez",
                "2",
                "Aetna, member 987654321, group 123456"
            ]
        },
        {
            "scenario": "📅 SCHEDULING - All Info at Once",
            "inputs": [
                "Hi, I'm Michael Johnson, DOB 12/10/1978, I live at 789 Pine Street, Chicago IL, email michael.j@yahoo.com, I want to book an appointment",
                "Dr. Emily Chen",
                "1", 
                "UnitedHealthcare, member 555666777, group 888999"
            ]
        },
        {
            "scenario": "❌ CANCELLATION - Natural",
            "inputs": [
                "Hi, I need to cancel my appointment",
                "I am Lisa Brown",
                "09/05/1995"
            ]
        },
        {
            "scenario": "❌ CANCELLATION - Full Info",
            "inputs": [
                "I want to cancel my appointment. I am David Wilson, DOB 04/18/1982"
            ]
        },
        {
            "scenario": "📅 SCHEDULING - Self Pay",
            "inputs": [
                "Hello, I want to book an appointment",
                "Amy Davis",
                "11/30/1988",
                "321 Elm Street, Boston, MA 02101", 
                "amy.davis@outlook.com",
                "Dr. Emily Chen",
                "3",
                "self-pay"
            ]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['scenario']}")
        print("-" * 40)
        for j, user_input in enumerate(test_case['inputs'], 1):
            print(f"   Step {j}: \"{user_input}\"")
        print()
    
    print("🎯 How to Test:")
    print("1. Copy any scenario's inputs")
    print("2. Paste them one by one into your chatbot")  
    print("3. See if the bot extracts names correctly")
    print("4. Check if the flow completes successfully")
    print()
    
    print("✅ Expected Results:")
    print("• Names extracted correctly (not 'I', 'am', etc.)")
    print("• Date formats normalized properly")  
    print("• Email addresses detected")
    print("• Appointments booked/cancelled successfully")
    print("• Emails sent (or simulated)")

def show_input_formats():
    """Show the correct input formats"""
    
    print("\n📋 CORRECT INPUT FORMATS")
    print("=" * 30)
    
    formats = {
        "Names": {
            "✅ Good": ["John Doe", "I am Sarah Smith", "My name is Michael Johnson"],
            "❌ Bad": ["okay", "sure", "yes", "I"]
        },
        "Dates": {
            "✅ Good": ["03/15/1985", "12/10/1978", "07/22/1990"],
            "❌ Bad": ["March 15th 1985", "15/03/1985", "1985"]
        },
        "Emails": {
            "✅ Good": ["john@gmail.com", "sarah.smith@yahoo.com", "user@company.co.uk"],
            "❌ Bad": ["john@", "invalid-email", "john.gmail.com"]
        },
        "Insurance": {
            "✅ Good": ["Blue Cross Blue Shield, member ID 123456789, group 987654", "Aetna", "self-pay"],
            "❌ Bad": ["insurance", "blue cross", "123456"]
        }
    }
    
    for category, examples in formats.items():
        print(f"\n{category}:")
        for status, items in examples.items():
            print(f"  {status}:")
            for item in items:
                print(f"    • \"{item}\"")

if __name__ == "__main__":
    test_scheduling_scenarios()
    show_input_formats()
    
    print("\n" + "=" * 60)
    print("🚀 Ready to test! Pick a scenario and try it with your agent!")
    print("=" * 60)