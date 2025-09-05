import os
import pandas as pd
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional, Tuple, Literal, Annotated
import uuid

# LangChain and LangGraph imports
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

# Corrected relative import
from .tools import SchedulingTools

class SchedulingState(TypedDict):
    """State class for the scheduling agent workflow"""
    messages: Annotated[List[BaseMessage], add_messages]
    current_stage: str
    intent: str  # 'schedule' or 'cancel'
    patient_info: Dict
    appointment_info: Dict
    insurance_info: Dict
    available_slots: List[Dict]
    appointment_id: Optional[str]

class AISchedulingAgent:
    """
    AI Scheduling Agent using LangGraph and LangChain with Groq integration
    """
    
    def __init__(self, groq_api_key: str):
        """Initialize the scheduling agent with LangGraph workflow"""
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.1-8b-instant",
            temperature=0.7,
            max_tokens=500
        )
        
        self.tools = SchedulingTools()
        self.workflow = self._build_workflow()
        print("✅ AI Scheduling Agent initialized with LangGraph + LangChain + Groq")

    def router(self, state: SchedulingState) -> str:
        """Router function to decide which node to run next."""
        
        # Initial routing - check if this is the first interaction
        if len(state["messages"]) <= 1:
            return "greeting"
            
        # Check for intent setting from greeting
        intent = state.get("intent", "")
        if intent == "cancel":
            return "cancellation"
        elif intent == "schedule":
            return state.get("current_stage", "patient_lookup")
            
        # Default routing based on current stage
        return state.get("current_stage", "greeting")

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for appointment scheduling"""
        workflow = StateGraph(SchedulingState)
        
        # Add all nodes
        workflow.add_node("greeting", self._greeting_node)
        workflow.add_node("patient_lookup", self._patient_lookup_node)
        workflow.add_node("smart_scheduling", self._smart_scheduling_node)
        workflow.add_node("calendar_integration", self._calendar_integration_node)
        workflow.add_node("insurance_collection", self._insurance_collection_node)
        workflow.add_node("appointment_confirmation", self._appointment_confirmation_node)
        workflow.add_node("form_distribution", self._form_distribution_node)
        workflow.add_node("cancellation", self._cancellation_node)
        workflow.add_node("end_conversation", self._end_conversation_node)
        
        # Add conditional routing from START
        workflow.add_conditional_edges(
            START, self.router, {
                "greeting": "greeting",
                "patient_lookup": "patient_lookup",
                "smart_scheduling": "smart_scheduling",
                "calendar_integration": "calendar_integration",
                "insurance_collection": "insurance_collection",
                "appointment_confirmation": "appointment_confirmation",
                "form_distribution": "form_distribution",
                "cancellation": "cancellation",
                "completed": "end_conversation"
            }
        )
        
        # Add edges to END for all nodes
        workflow.add_edge("greeting", END)
        workflow.add_edge("patient_lookup", END)
        workflow.add_edge("smart_scheduling", END)
        workflow.add_edge("calendar_integration", END)
        workflow.add_edge("insurance_collection", END)
        workflow.add_edge("appointment_confirmation", END)
        workflow.add_edge("form_distribution", END)
        workflow.add_edge("cancellation", END)
        workflow.add_edge("end_conversation", END)
        
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    def _greeting_node(self, state: SchedulingState) -> Dict:
        """Node 1: Initial greeting and intent detection"""
        
        # Initial greeting
        if len(state["messages"]) <= 1:
            greeting_message = """Hello! Welcome to HealthCare Plus Medical Center. 🏥

I'm your AI scheduling assistant. I can help you with:
• **Scheduling a new appointment**
• **Canceling an existing appointment**

How can I assist you today?"""
            
            return {
                "messages": [AIMessage(content=greeting_message)], 
                "current_stage": "greeting"
            }
        
        # Intent detection from user response
        last_message = state["messages"][-1]
        if isinstance(last_message, HumanMessage):
            message_content = last_message.content.lower()
            
            # Check for cancellation intent
            cancel_keywords = ["cancel", "cancellation", "remove", "delete", "reschedule"]
            if any(keyword in message_content for keyword in cancel_keywords):
                response = "I'll help you cancel your appointment. Let me gather some information first."
                return {
                    "messages": [AIMessage(content=response)],
                    "current_stage": "cancellation",
                    "intent": "cancel"
                }
            
            # Check for scheduling intent
            schedule_keywords = ["schedule", "book", "appointment", "new", "visit", "see doctor", "make", "like"]
            if any(keyword in message_content for keyword in schedule_keywords):
                response = "Great! I'll help you schedule an appointment. Let me start by getting some information. \nWhat is your First Name?"
                return {
                    "messages": [AIMessage(content=response)],
                    "current_stage": "patient_lookup",
                    "intent": "schedule"
                }
            
            # Clarification needed
            response = """I'd be happy to help! Could you please specify if you'd like to:

1. **Schedule a new appointment**
2. **Cancel an existing appointment**

Please let me know which option you prefer."""
            
            return {
                "messages": [AIMessage(content=response)],
                "current_stage": "greeting"
            }

    def _patient_lookup_node(self, state: SchedulingState) -> Dict:
        """Node 2: Patient Information Collection and Lookup - SIMPLE & RELIABLE VERSION"""
        
        last_message = state["messages"][-1]
        current_patient_info = state.get("patient_info", {})
        
        # Extract patient information from the latest message
        if isinstance(last_message, HumanMessage) and last_message.content.strip():
            user_input = last_message.content.strip()
            
            # Check what field we're currently asking for based on missing fields
            required_fields = ["first_name", "last_name", "dob", "location", "email"]
            missing_fields = [field for field in required_fields if not current_patient_info.get(field)]
            
            if missing_fields:
                current_field = missing_fields[0]
                
                # SIMPLE & RELIABLE EXTRACTION
                if current_field == "first_name":
                    # Extract names using simple but effective method
                    first_name, last_name = self._simple_name_extraction(user_input)
                    
                    if first_name:
                        current_patient_info["first_name"] = first_name
                        if last_name:  # Got both names at once
                            current_patient_info["last_name"] = last_name
                            print(f"✅ Extracted both names: {first_name} {last_name}")
                    else:
                        # Check if it's a non-informative response
                        non_name_responses = ["okay", "sure", "yes", "no", "ok", "yeah", "yep", "alright", "fine"]
                        if user_input.lower() not in non_name_responses:
                            # Use LLM for complex cases
                            extracted_names = self._llm_name_extraction(user_input)
                            if extracted_names:
                                current_patient_info.update(extracted_names)
                            else:
                                # Fallback: take the input as-is
                                current_patient_info["first_name"] = user_input
                        else:
                            print(f"⚠️ Ignoring non-informative response: '{user_input}'")
                        
                elif current_field == "last_name":
                    # Simple last name extraction
                    if user_input and len(user_input.strip()) > 0:
                        current_patient_info["last_name"] = user_input.strip()
                        
                elif current_field == "dob":
                    try:
                        current_patient_info["dob"] = self.tools._normalize_date_format(user_input)
                    except:
                        current_patient_info["dob"] = user_input
                        
                elif current_field == "location":
                    current_patient_info["location"] = user_input.strip()
                    
                elif current_field == "email":
                    # Extract email if it contains @ symbol
                    import re
                    email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', user_input)
                    if email_match:
                        current_patient_info["email"] = email_match.group()
                    else:
                        current_patient_info["email"] = user_input.strip()
                
                print(f"✅ Simple extraction - Field: {current_field}, Input: '{user_input}', Value: {current_patient_info.get(current_field)}")
                print(f"✅ Current patient info: {current_patient_info}")
            
        # Check for required fields and ask for missing ones
        required_fields = ["first_name", "last_name", "dob", "location", "email"]
        missing_fields = [field for field in required_fields if not current_patient_info.get(field)]
        
        if missing_fields:
            # Ask for the next missing field
            field_questions = {
                "first_name": "What is your first name?",
                "last_name": "What is your last name?", 
                "dob": "What is your date of birth? Please use MM/DD/YYYY format.",
                "location": "What is your home address?",
                "email": "What is your email address?"
            }
            
            next_field = missing_fields[0]
            question = field_questions.get(next_field)
            
            # Provide progress feedback
            completed_fields = [f for f in required_fields if current_patient_info.get(f)]
            if completed_fields:
                completed_names = [f.replace('_', ' ').title() for f in completed_fields]
                progress = f"Got it! ✅ **{', '.join(completed_names)}**\n\n"
            else:
                progress = ""
                
            response = f"{progress}{question}"
            stage = "patient_lookup"
        else:
            # All required info collected, perform patient lookup
            is_returning = self.tools.lookup_patient(
                current_patient_info.get("first_name", ""),
                current_patient_info.get("last_name", ""), 
                current_patient_info.get("dob", "")
            )
            
            current_patient_info["is_returning"] = is_returning
            patient_type = "returning patient" if is_returning else "new patient"
            
            response = f"""Perfect! I found you in our system as a **{patient_type}**, {current_patient_info.get('first_name', '')}.

**Your Information:**
• **Name:** {current_patient_info.get('first_name')} {current_patient_info.get('last_name')}
• **DOB:** {current_patient_info.get('dob')}
• **Email:** {current_patient_info.get('email')}

Which doctor would you prefer for your appointment?
• **Dr. Emily Chen** (Internal Medicine)
• **Dr. David Rodriguez** (Family Practice)

Please select your preferred doctor."""
            stage = "smart_scheduling"
        
        return {
            "messages": [AIMessage(content=response)],
            "current_stage": stage,
            "patient_info": current_patient_info
        }

    def _simple_name_extraction(self, text: str) -> tuple:
        """Simple but reliable name extraction"""
        import re
        
        # Clean the text first
        text = text.strip()
        
        # Pattern 1: "I am FirstName LastName" (most common)
        match = re.search(r'\bi am\s+([A-Z][a-z]+)(?:\s+([A-Z][a-z]*))?\b', text, re.IGNORECASE)
        if match:
            first_name = match.group(1)
            last_name = match.group(2) if match.group(2) else None
            return first_name, last_name
        
        # Pattern 2: "My name is FirstName LastName"
        match = re.search(r'\bmy name is\s+([A-Z][a-z]+)(?:\s+([A-Z][a-z]*))?\b', text, re.IGNORECASE)
        if match:
            first_name = match.group(1)
            last_name = match.group(2) if match.group(2) else None
            return first_name, last_name
        
        # Pattern 3: "This is FirstName LastName"
        match = re.search(r'\bthis is\s+([A-Z][a-z]+)(?:\s+([A-Z][a-z]*))?\b', text, re.IGNORECASE)
        if match:
            first_name = match.group(1)
            last_name = match.group(2) if match.group(2) else None
            return first_name, last_name
        
        # Pattern 4: Just "FirstName LastName" (two capitalized words)
        words = text.split()
        if len(words) >= 2:
            # Look for two consecutive capitalized words that look like names
            for i in range(len(words) - 1):
                word1, word2 = words[i], words[i + 1]
                if (self._looks_like_name(word1) and self._looks_like_name(word2) and 
                    not self._is_common_word(word1) and not self._is_common_word(word2)):
                    return word1, word2
        
        return None, None

    def _looks_like_name(self, word: str) -> bool:
        """Check if a word looks like it could be a name"""
        if not word or len(word) < 2:
            return False
        # Should start with capital letter and contain only letters
        return word[0].isupper() and word.isalpha()

    def _is_common_word(self, word: str) -> bool:
        """Check if a word is a common word that's not a name"""
        common_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 
            'after', 'above', 'below', 'between', 'among', 'this', 'that', 'these', 
            'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 
            'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their',
            'like', 'want', 'need', 'book', 'cancel', 'schedule', 'appointment'
        }
        return word.lower() in common_words

    def _llm_name_extraction(self, text: str) -> dict:
        """Use LLM for complex name extraction as fallback"""
        try:
            extraction_prompt = ChatPromptTemplate.from_template("""
Extract ONLY the first name and last name from this text. 
Return a JSON object with "first_name" and "last_name" keys.
If you cannot find clear names, return empty strings.

Examples:
"I am John Smith" -> {{"first_name": "John", "last_name": "Smith"}}
"My name is Sarah" -> {{"first_name": "Sarah", "last_name": ""}}
"Hello there" -> {{"first_name": "", "last_name": ""}}

Text: "{message}"
""")
            
            parser = JsonOutputParser()
            extraction_chain = extraction_prompt | self.llm | parser
            result = extraction_chain.invoke({"message": text})
            
            if isinstance(result, dict):
                # Only return if we have at least a first name
                first_name = result.get("first_name", "").strip()
                last_name = result.get("last_name", "").strip()
                
                if first_name and len(first_name) > 1 and not self._is_common_word(first_name):
                    return {"first_name": first_name, "last_name": last_name} if last_name else {"first_name": first_name}
            
        except Exception as e:
            print(f"❌ LLM extraction failed: {e}")
        
        return None
    def _extract_names_from_text(self, text: str) -> tuple:
        """Extract first and last names from natural language text"""
        import re
        
        # Common patterns for name extraction
        patterns = [
            # "I am John Doe" or "I'm John Doe"
            r'\b(?:i am|i\'m)\s+([A-Z][a-z]+)\s+([A-Z][a-z]+)\b',
            # "My name is John Doe"
            r'\bmy name is\s+([A-Z][a-z]+)\s+([A-Z][a-z]+)\b',
            # "This is John Doe"
            r'\bthis is\s+([A-Z][a-z]+)\s+([A-Z][a-z]+)\b',
            # "John Doe from" or "John Doe here"
            r'\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\s+(?:from|here|speaking)\b',
            # Just "John Doe" (two capitalized words)
            r'\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                first_name, last_name = match.groups()
                # Validate that these look like real names (not "I" "am", etc.)
                if self._is_valid_name(first_name) and self._is_valid_name(last_name):
                    return first_name.title(), last_name.title()
        
        return None, None
    
    def _is_valid_name(self, name: str) -> bool:
        """Check if a string looks like a valid name"""
        if not name or len(name) < 2:
            return False
        
        # Common words that are NOT names
        non_names = ["from", "the", "and", "with", "here", "there", "this", "that", "am", "is", "was", "were"]
        
        return name.lower() not in non_names
    
    def _extract_location_from_text(self, text: str) -> str:
        """Extract location from natural language text"""
        import re
        
        # Patterns like "I am from New York" or "from Chennai"
        location_patterns = [
            r'\bfrom\s+(.+?)(?:\s|$)',
            r'\bin\s+([A-Z][a-zA-Z\s,]+?)(?:\s|$)',
            r'\bat\s+([A-Z][a-zA-Z\s,]+?)(?:\s|$)'
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                location = match.group(1).strip()
                # Clean up common trailing words
                location = re.sub(r'\s+(from|in|at)$', '', location, flags=re.IGNORECASE)
                return location
        
        # If no pattern matches, return the text as-is
        return text
    
    def _has_poor_quality_names(self, patient_info: dict) -> bool:
        """Check if the extracted names look like poor quality (e.g., 'I', 'am', etc.)"""
        first_name = patient_info.get('first_name', '')
        last_name = patient_info.get('last_name', '')
        
        poor_quality_indicators = [
            len(first_name) <= 1,  # Single character names
            len(last_name) <= 1,
            first_name.lower() in ['i', 'am', 'my', 'me', 'the', 'from', 'in', 'at'],
            last_name.lower() in ['i', 'am', 'my', 'me', 'the', 'from', 'in', 'at']
        ]
        
        return any(poor_quality_indicators)
    
    def _extract_names_from_all_messages(self, messages: list) -> tuple:
        """Look through all messages to find proper names"""
        for message in messages:
            if isinstance(message, HumanMessage):
                first_name, last_name = self._extract_names_from_text(message.content)
                if first_name and last_name:
                    return first_name, last_name
        return None, None

    def _smart_scheduling_node(self, state: SchedulingState) -> Dict:
        """Node 3: Smart Scheduling based on patient type and doctor preference"""
        
        patient_info = state["patient_info"]
        appointment_info = state.get("appointment_info", {})
        last_message = state["messages"][-1]
        
        # Determine appointment duration based on patient type
        appointment_duration = 30 if patient_info.get("is_returning") else 60
        appointment_info["duration"] = appointment_duration
        
        # Extract doctor preference
        doctor = appointment_info.get("doctor")
        if not doctor and isinstance(last_message, HumanMessage):
            message_content = last_message.content.lower()
            if 'emily' in message_content or 'chen' in message_content:
                doctor = 'Dr. Emily Chen'
            elif 'david' in message_content or 'rodriguez' in message_content:
                doctor = 'Dr. David Rodriguez'
        
        if doctor:
            appointment_info["doctor_name"] = doctor
            patient_type = "returning" if patient_info.get("is_returning") else "new"
            
            response = f"""Excellent choice! I'm scheduling a **{appointment_duration}-minute appointment** with **{doctor}** for a {patient_type} patient.

Let me check their availability..."""
            stage = "calendar_integration"
        else:
            response = "Please select one of the available doctors to continue with your appointment."
            stage = "smart_scheduling"
        
        return {
            "messages": [AIMessage(content=response)],
            "current_stage": stage,
            "appointment_info": appointment_info
        }

    def _calendar_integration_node(self, state: SchedulingState) -> Dict:
        """Node 4: Show available slots and handle booking"""
        
        appointment_info = state.get("appointment_info", {})
        last_message = state["messages"][-1]
        
        # Handle slot selection
        if isinstance(last_message, HumanMessage):
            try:
                slot_number = int(last_message.content.strip()) - 1
                available_slots = state.get("available_slots", [])
                
                if 0 <= slot_number < len(available_slots):
                    selected_slot = available_slots[slot_number]
                    appointment_info.update(selected_slot)
                    
                    response = f"""Perfect! You've selected:

**Doctor:** {selected_slot['doctor_name']}
**Date:** {selected_slot['date']}
**Time:** {selected_slot['time']}

Now I need to collect your insurance information to complete the booking."""
                    
                    return {
                        "messages": [AIMessage(content=response)],
                        "current_stage": "insurance_collection",
                        "appointment_info": appointment_info
                    }
                else:
                    response = f"Please choose a valid slot number between 1 and {len(available_slots)}."
                    return {
                        "messages": [AIMessage(content=response)],
                        "current_stage": "calendar_integration"
                    }
                    
            except (ValueError, IndexError):
                pass
        
        # Get and display available slots
        available_slots = self.tools.get_available_slots(
            doctor=appointment_info.get("doctor_name"),
            duration=appointment_info.get("duration", 30)
        )
        
        if available_slots:
            slots_text = "\n".join([
                f"**{i+1}.** {s['doctor_name']} - {s['date']} at {s['time']}"
                for i, s in enumerate(available_slots)
            ])
            
            response = f"""Here are the available appointment slots:

{slots_text}

Please select a slot by entering the number (1-{len(available_slots)})."""
            stage = "calendar_integration"
        else:
            response = """I'm sorry, there are no available slots for the selected doctor right now. 

Would you like to:
1. Try the other doctor
2. Check different dates

Please let me know your preference."""
            stage = "smart_scheduling"
            appointment_info.pop("doctor_name", None)
        
        return {
            "messages": [AIMessage(content=response)],
            "current_stage": stage,
            "available_slots": available_slots,
            "appointment_info": appointment_info
        }

    def _insurance_collection_node(self, state: SchedulingState) -> Dict:
        """Node 5: Insurance Collection with self-pay option"""
        
        insurance_info = state.get("insurance_info", {})
        last_message = state["messages"][-1]
        
        if isinstance(last_message, HumanMessage):
            message_content = last_message.content.lower()
            
            # Check for self-pay
            self_pay_keywords = ["no insurance", "self pay", "self-pay", "i don't have", "paying myself", "cash", "no"]
            if any(keyword in message_content for keyword in self_pay_keywords):
                insurance_info.update({
                    "carrier": "Self-Pay",
                    "member_id": "N/A", 
                    "group_number": "N/A"
                })
                response = "Understood. I've marked you as a **self-pay patient**. Let me confirm your appointment details now."
                return {
                    "messages": [AIMessage(content=response)],
                    "current_stage": "appointment_confirmation",
                    "insurance_info": insurance_info
                }
            
            # Extract insurance information
            try:
                prompt = ChatPromptTemplate.from_template("""
Extract insurance details from the message.
Return JSON with keys: "carrier", "member_id", "group_number".
Use empty string "" if missing.

Examples:
"Blue Cross Blue Shield, member ID 123456789, group 987654" -> {{"carrier": "Blue Cross Blue Shield", "member_id": "123456789", "group_number": "987654"}}
"Aetna insurance" -> {{"carrier": "Aetna", "member_id": "", "group_number": ""}}

Extract from: "{message}"
""")
                
                parser = JsonOutputParser()
                chain = prompt | self.llm | parser
                extracted = chain.invoke({"message": last_message.content})
                
                if isinstance(extracted, dict):
                    # Only update with non-empty values
                    for key, value in extracted.items():
                        if value and value.strip():
                            insurance_info[key] = value.strip()
            except Exception as e:
                print(f"❌ Error extracting insurance info: {e}")
        
        # Check for required information
        required = ["carrier", "member_id", "group_number"]
        missing = [field for field in required if not insurance_info.get(field)]
        
        if not missing:
            response = "Thank you! I have all your insurance information. Let me confirm your appointment details."
            stage = "appointment_confirmation"
        else:
            if not insurance_info:
                response = """Please provide your insurance information, or type "self-pay" if you don't have insurance:

• **Insurance Carrier** (e.g., Blue Cross Blue Shield, Aetna)
• **Member ID**
• **Group Number**

You can provide all details at once or type "self-pay" if paying out of pocket."""
            else:
                missing_readable = missing[0].replace('_', ' ').title()
                response = f"Please provide your **{missing_readable}**."
            stage = "insurance_collection"
        
        return {
            "messages": [AIMessage(content=response)],
            "current_stage": stage,
            "insurance_info": insurance_info
        }

    def _appointment_confirmation_node(self, state: SchedulingState) -> Dict:
        """Node 6: Appointment Confirmation and Booking"""
        
        patient_info = state["patient_info"]
        appointment_info = state["appointment_info"] 
        insurance_info = state["insurance_info"]
        
        # Save the appointment
        appointment_id = self.tools.save_appointment(patient_info, appointment_info, insurance_info)
        
        confirmation_response = f"""🎉 **APPOINTMENT CONFIRMED** 🎉

**Appointment Details:**
• **Patient:** {patient_info.get('first_name')} {patient_info.get('last_name')}
• **Doctor:** {appointment_info.get('doctor_name')}
• **Date & Time:** {appointment_info.get('date')} at {appointment_info.get('time')}
• **Duration:** {appointment_info.get('duration')} minutes
• **Insurance:** {insurance_info.get('carrier')}
• **Appointment ID:** {appointment_id}

✅ Your appointment has been successfully booked!
📧 You'll receive a confirmation email shortly with all the details."""
        
        return {
            "messages": [AIMessage(content=confirmation_response)],
            "current_stage": "form_distribution",
            "appointment_id": appointment_id
        }

    def _form_distribution_node(self, state: SchedulingState) -> Dict:
        """Node 7: Form Distribution for new patients (only after confirmation)"""
        
        patient_info = state["patient_info"]
        
        if not patient_info.get("is_returning", True):  # New patient
            email = patient_info.get('email')
            if email:
                sent = self.tools.send_patient_intake_form(
                    email, 
                    f"{patient_info['first_name']} {patient_info['last_name']}"
                )
                
                if sent:
                    form_response = """📋 **New Patient Forms**

As a new patient, I've sent the intake form to your email address. Please:
• Complete the form before your appointment
• Bring it with you or submit it online
• Arrive 15 minutes early for check-in"""
                else:
                    form_response = """📋 **New Patient Forms**

I tried to send your intake form, but there was an issue with the email delivery.
Please contact our office at (555) 123-4567 to receive your forms, or arrive 15 minutes early to complete them at the clinic."""
            else:
                form_response = """📋 **New Patient Forms**

As a new patient, you'll need to complete intake forms. Please arrive 15 minutes early to fill them out, or contact our office at (555) 123-4567 to receive them in advance."""
        else:  # Returning patient
            form_response = "As a returning patient, no additional forms are needed. Just arrive on time for your appointment!"
        
        final_message = f"""{form_response}

🔔 **Reminder System**
You'll receive appointment reminders via email and SMS before your visit.

Is there anything else I can help you with today?"""
        
        return {
            "messages": [AIMessage(content=final_message)],
            "current_stage": "completed"
        }

    def _cancellation_node(self, state: SchedulingState) -> Dict:
        """Enhanced cancellation node with better user experience"""
        
        patient_info = state.get("patient_info", {})
        last_message = state["messages"][-1]
        
        if isinstance(last_message, HumanMessage):
            # Context-aware extraction for cancellation too
            required_fields = ["first_name", "last_name", "dob"]
            missing_fields = [field for field in required_fields if not patient_info.get(field)]
            
            if missing_fields:
                current_field = missing_fields[0]
                user_input = last_message.content.strip()
                
                # Direct assignment based on context
                if current_field == "first_name":
                    names = user_input.split()
                    patient_info["first_name"] = names[0] if names else user_input
                elif current_field == "last_name":
                    names = user_input.split()
                    patient_info["last_name"] = names[-1] if names else user_input
                elif current_field == "dob":
                    try:
                        patient_info["dob"] = self.tools._normalize_date_format(user_input)
                    except:
                        patient_info["dob"] = user_input
                        
                print(f"✅ Cancellation field {current_field}: {patient_info.get(current_field)}")
        
        # Check for required information
        required_fields = ["first_name", "last_name", "dob"]
        missing_fields = [field for field in required_fields if not patient_info.get(field)]
        
        if missing_fields:
            field_questions = {
                "first_name": "What is your first name?",
                "last_name": "What is your last name?", 
                "dob": "What is your date of birth? Please use MM/DD/YYYY format."
            }
            
            if not patient_info:
                question = "To cancel your appointment, I need to verify your identity. What is your first name?"
            else:
                question = field_questions.get(missing_fields[0])
            
            return {
                "messages": [AIMessage(content=question)],
                "current_stage": "cancellation",
                "patient_info": patient_info
            }
        
        # Look for the appointment to cancel
        appointment_to_cancel = self.tools.find_appointment_by_patient(patient_info)
        
        if not appointment_to_cancel:
            response = f"""I couldn't find an active appointment for **{patient_info['first_name']} {patient_info['last_name']}**.

This could be because:
• The appointment was already cancelled
• The name or date of birth doesn't match our records
• There might be a spelling difference

Would you like to try again with different information, or would you prefer to call our office at (555) 123-4567 for assistance?"""
            
            return {
                "messages": [AIMessage(content=response)],
                "current_stage": "completed"
            }
        
        # Cancel the appointment
        appointment_id = appointment_to_cancel['appointment_id']
        cancelled = self.tools.cancel_appointment(
            appointment_id, 
            reason="Patient requested cancellation via AI assistant"
        )
        
        if cancelled:
            response = f"""✅ **Appointment Successfully Cancelled**

**Cancelled Appointment Details:**
• **Patient:** {appointment_to_cancel['patient_first_name']} {appointment_to_cancel['patient_last_name']}
• **Doctor:** {appointment_to_cancel['doctor_name']}
• **Date & Time:** {appointment_to_cancel['appointment_date']} at {appointment_to_cancel['appointment_time']}
• **Appointment ID:** {appointment_id}

Your appointment slot has been freed up for other patients. If you'd like to reschedule, I can help you book a new appointment right away!

Would you like to schedule a new appointment now?"""
        else:
            response = f"""❌ **Cancellation Error**

I'm sorry, but I was unable to cancel appointment **{appointment_id}**. This might be due to a system issue.

Please contact our office directly at:
📞 **(555) 123-4567**
📧 **appointments@healthcareplus.com**

Our staff will be happy to assist you with the cancellation."""
        
        return {
            "messages": [AIMessage(content=response)],
            "current_stage": "completed"
        }

    def _end_conversation_node(self, state: SchedulingState) -> Dict:
        """Final node to end conversation gracefully"""
        return {
            "messages": [],
            "current_stage": "completed"
        }

    def process_message(self, user_message: str, thread_id: str = "default") -> str:
        """Process user message through the LangGraph workflow"""
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            # Handle initial conversation start
            if user_message == "start conversation":
                user_msg = HumanMessage(content="Hello")
            else:
                user_msg = HumanMessage(content=user_message)
            
            result = self.workflow.invoke({"messages": [user_msg]}, config=config)
            
            # Extract the latest AI message
            ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage) and msg.content.strip()]
            
            if ai_messages:
                return ai_messages[-1].content
            else:
                return "I'm here to help! How can I assist you with scheduling or canceling an appointment today?"
                
        except Exception as e:
            print(f"❌ LangGraph workflow error: {e}")
            return "I'm experiencing technical difficulties. Please try again, or contact our office at (555) 123-4567 for assistance."

    def get_workflow_state(self, thread_id: str = "default") -> Dict:
        """Get current workflow state for debugging"""
        try:
            state = self.workflow.get_state({"configurable": {"thread_id": thread_id}})
            if state and 'values' in state:
                # Convert messages to JSON for serialization
                if 'messages' in state['values']:
                    state['values']['messages'] = [
                        msg.dict() if hasattr(msg, 'dict') else str(msg) 
                        for msg in state['values']['messages']
                    ]
                return state['values']
            return {}
        except Exception as e:
            print(f"❌ Error getting workflow state: {e}")
            return {}

    def reset_conversation(self, thread_id: str = "default") -> bool:
        """Reset conversation state for a new session"""
        try:
            config = {"configurable": {"thread_id": thread_id}}
            self.workflow.update_state(config, {
                "messages": [],
                "current_stage": "greeting",
                "intent": "",
                "patient_info": {},
                "appointment_info": {},
                "insurance_info": {},
                "available_slots": [],
                "appointment_id": None
            })
            return True
        except Exception as e:
            print(f"❌ Error resetting conversation: {e}")
            return False