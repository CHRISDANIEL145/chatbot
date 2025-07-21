import os
import secrets
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import json
import uuid
import re
from datetime import datetime
from PyPDF2 import PdfReader

app = Flask(__name__)
CORS(app)

# --- Gemini API Configuration ---
API_KEY = 'AIzaSyClybNYiaFXMEQSBdCYzW6zsi9qvIdQiyc' # Replace with your actual key
if not API_KEY or API_KEY == 'YOUR_GEMINI_API_KEY_HERE':
    raise ValueError("GEMINI_API_KEY is not set. Please replace 'YOUR_GEMINI_API_KEY_HERE' in app.py with your actual key.")
genai.configure(api_key=API_KEY)

# --- Global State (in-memory session management) ---
sessions = {}

def get_or_create_session(session_id):
    if session_id not in sessions:
        sessions[session_id] = {
            'candidate_profile': None,
            'interview_questions': [],
            'interview_responses': [],
            'interview_start_time': None,
            'interview_end_time': None
        }
    return sessions[session_id]

# --- AI Models ---
try:
    resume_analyzer_model = genai.GenerativeModel('gemini-1.5-flash')
    question_generator_model = genai.GenerativeModel('gemini-1.5-flash')
    response_evaluator_model = genai.GenerativeModel('gemini-1.5-flash')
    assessment_generator_model = genai.GenerativeModel('gemini-1.5-flash')
    # text_embedding_model = genai.GenerativeModel('embedding-001') # Not used in current functionality
    print("Gemini models loaded successfully.")
except Exception as e:
    print(f"Error loading Gemini models: {e}")
    # Consider more graceful error handling for production, e.g., exit or disable AI features.

# --- Utility Functions ---

def extract_text_from_pdf(pdf_file):
    """
    Extracts text from an uploaded PDF file using PyPDF2.
    Returns the extracted text as a string.
    """
    text = ""
    try:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def generate_content_with_gemini(model, prompt, **kwargs):
    """Helper to generate content safely with Gemini, including error handling and token count."""
    try:
        token_count_response = model.count_tokens(prompt)
        print(f"DEBUG: Prompt token count: {token_count_response.total_tokens} tokens.")
        
        # Adding safety settings and generation config for JSON
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        
        # For structured outputs, consider setting response_mime_type if the model supports it well.
        # For gemini-1.5-flash, direct prompt engineering and robust parsing are still key.
        generation_config = {
            "response_mime_type": "application/json" # This is a hint, not a guarantee for all models
        }
        
        response = model.generate_content(
            prompt, 
            safety_settings=safety_settings, 
            generation_config=generation_config,
            **kwargs
        )
        
        if response and response.text:
            return response.text
        else:
            print("DEBUG: Gemini response was empty or malformed.")
            return None
    except Exception as e:
        print(f"DEBUG: Error calling Gemini API: {e}")
        # Log the full error to help debug API issues
        if hasattr(e, 'response') and hasattr(e.response, 'json'):
            print(f"DEBUG: Gemini API full error response: {e.response.json()}")
        return None

def extract_json_from_gemini_response(text, is_array=False):
    """
    Extracts a JSON string from a potentially noisy Gemini text response.
    Handles cases with and without ```json``` block.
    If is_array is True, it tries to find an array. Otherwise, it finds an object.
    """
    if not text:
        return None

    # Try to find a JSON block wrapped in ```json ... ```
    json_match = re.search(r'```json\n(.*)\n```', text, re.DOTALL)
    if json_match:
        return json_match.group(1).strip()

    # Fallback: Try to find a JSON object or array directly in the text
    if is_array:
        # Matches the first array from '[' to ']'
        direct_json_match = re.search(r'\[.*\]', text, re.DOTALL)
    else:
        # Matches the first object from '{' to '}'
        direct_json_match = re.search(r'{.*}', text, re.DOTALL)
    
    if direct_json_match:
        return direct_json_match.group(0).strip()
    
    # If no specific JSON structure is found, return the whole text as a last resort
    return text.strip()


# --- API Endpoints ---

@app.route('/upload_resume', methods=['POST'])
def upload_resume():
    session_id = request.headers.get('X-User-Session-Id', str(uuid.uuid4()))
    session = get_or_create_session(session_id)

    if 'resume' not in request.files:
        return jsonify({'error': 'No resume file provided'}), 400
    
    file = request.files['resume']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        resume_content = extract_text_from_pdf(file)
        
        if not resume_content.strip():
            return jsonify({'error': 'Could not extract text from the provided PDF. Please ensure it is a text-based PDF or its text is extractable.'}), 400

        prompt = f"""Analyze the following resume text and extract the candidate's name, email, total years of experience (if quantifiable, otherwise a brief summary like "2 roles (5 years)"), a list of key skills, and an inferred primary job role/position.
        
        Ensure the 'key_skills' is always a JSON array of strings, even if empty.
        
        Format the output strictly as a JSON object with the following keys: `name` (string), `email` (string), `experience` (string, e.g., "5 years" or "2 roles (5 years)"), `key_skills` (array of strings), `inferred_position` (string).
        
        Example JSON output:
        ```json
        {{
          "name": "John Doe",
          "email": "john.doe@example.com",
          "experience": "5 years",
          "key_skills": ["Python", "Machine Learning", "Data Science"],
          "inferred_position": "Data Scientist"
        }}
        ```

        Resume Text:
        ---
        {resume_content}
        ---
        """
        
        try:
            gemini_response_text = generate_content_with_gemini(resume_analyzer_model, prompt)
            
            if gemini_response_text:
                print(f"DEBUG: Raw Gemini response for resume_analyzer: {gemini_response_text}")
                cleaned_response = extract_json_from_gemini_response(gemini_response_text, is_array=False)
                
                if not cleaned_response:
                     raise ValueError("Failed to extract valid JSON from Gemini response for resume analysis.")

                candidate_profile = json.loads(cleaned_response)
                
                # Ensure key_skills is always a list
                if not isinstance(candidate_profile.get('key_skills'), list):
                    if isinstance(candidate_profile.get('key_skills'), str):
                        candidate_profile['key_skills'] = [s.strip() for s in candidate_profile['key_skills'].split(',') if s.strip()]
                    else:
                        candidate_profile['key_skills'] = []
                
                session['candidate_profile'] = candidate_profile
                return jsonify({'message': 'Resume processed successfully', 'candidate_profile': candidate_profile, 'session_id': session_id}), 200
            else:
                return jsonify({'error': 'AI failed to parse resume or returned empty response.'}), 500

        except json.JSONDecodeError as e:
            print(f"DEBUG: JSON Decode Error in /upload_resume: {e}")
            print(f"DEBUG: Raw AI response that caused error: {gemini_response_text}")
            return jsonify({'error': f'Failed to parse AI response as JSON for resume upload: {e}. Raw AI response: {gemini_response_text}'}), 500
        except ValueError as e: # Catch our custom ValueError for extraction issues
             print(f"DEBUG: Value Error in /upload_resume: {e}")
             return jsonify({'error': f'{str(e)}. Raw AI response: {gemini_response_text}'}), 500
        except Exception as e:
            print(f"DEBUG: Error during resume processing in /upload_resume: {e}")
            return jsonify({'error': f'An unexpected error occurred during AI processing: {str(e)}'}), 500

@app.route('/setup_interview', methods=['POST'])
def setup_interview():
    session_id = request.headers.get('X-User-Session-Id')
    if not session_id or session_id not in sessions:
        return jsonify({'error': 'Invalid or missing session ID'}), 400
    session = sessions[session_id]

    data = request.get_json()
    position_role = data.get('position_role')
    candidate_profile = session.get('candidate_profile') 

    if not position_role or not candidate_profile:
        return jsonify({'error': 'Position role and candidate profile are required'}), 400
    
    session['candidate_profile']['position'] = position_role

    skills = ", ".join(candidate_profile.get('key_skills', []))
    experience = candidate_profile.get('experience', 'N/A')
    candidate_name = candidate_profile.get('name', 'Candidate')

    prompt = f"""As an expert interviewer, generate interview questions for a candidate named {candidate_name} applying for a '{position_role}' role.
    The candidate has {experience} of experience and these key skills: {skills}.
    
    Generate the following specific number of questions:
    - 10 Technical questions
    - 3 Soft Skills questions
    - 2 Communication Skills questions

    For each question, also provide 1-3 relevant tags (e.g., 'technical', 'experience', 'soft skills', 'problem-solving', 'leadership', 'communication', 'project').
    
    Format the output strictly as a JSON array of objects. Each object should have the following keys:
    - `id`: A unique string ID for the question.
    - `question`: The interview question.
    - `tags`: An array of strings representing the tags.
    
    Example JSON format:
    ```json
    [
      {{
        "id": "q1",
        "question": "Can you describe a challenging project you worked on and how you overcame obstacles?",
        "tags": ["experience", "problem-solving"]
      }},
      {{
        "id": "q2",
        "question": "Explain the concept of RESTful APIs and how you've used them in your projects.",
        "tags": ["technical", "api"]
      }}
    ]
    ```
    """
    
    try:
        gemini_response_text = generate_content_with_gemini(question_generator_model, prompt)

        if gemini_response_text:
            print(f"DEBUG: Raw Gemini response for question_generator: {gemini_response_text}")
            cleaned_response = extract_json_from_gemini_response(gemini_response_text, is_array=True)
            
            if not cleaned_response:
                raise ValueError("Failed to extract valid JSON from Gemini response for question generation.")

            questions = json.loads(cleaned_response)
            session['interview_questions'] = questions
            session['interview_responses'] = []
            session['interview_start_time'] = datetime.now().isoformat()
            return jsonify({'message': 'Interview questions generated', 'questions': questions}), 200
        else:
            return jsonify({'error': 'AI failed to generate questions or returned empty response.'}), 500

    except json.JSONDecodeError as e:
        print(f"DEBUG: JSON Decode Error in /setup_interview: {e}")
        print(f"DEBUG: Raw AI response that caused error: {gemini_response_text}")
        return jsonify({'error': f'Failed to parse AI response as JSON for interview setup: {e}. Raw AI response: {gemini_response_text}'}), 500
    except ValueError as e:
        print(f"DEBUG: Value Error in /setup_interview: {e}")
        return jsonify({'error': f'{str(e)}. Raw AI response: {gemini_response_text}'}), 500
    except Exception as e:
        print(f"DEBUG: Error during question generation in /setup_interview: {e}")
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

@app.route('/submit_answer', methods=['POST'])
def submit_answer():
    session_id = request.headers.get('X-User-Session-Id')
    if not session_id or session_id not in sessions:
        return jsonify({'error': 'Invalid or missing session ID'}), 400
    session = sessions[session_id]

    data = request.get_json()
    question_id = data.get('question_id')
    response_text = data.get('response_text')
    duration = data.get('duration')

    if not question_id or not response_text:
        return jsonify({'error': 'Question ID and response text are required'}), 400

    question_obj = next((q for q in session['interview_questions'] if q['id'] == question_id), None)
    if not question_obj:
        return jsonify({'error': 'Question not found in current session'}), 404

    prompt = f"""You are an AI interviewer. Evaluate the following candidate's response to an interview question.
    Provide a score out of 100 for Technical accuracy, Communication clarity, and Relevance to the question.
    Also, provide a brief feedback on the response.
    
    Format the output strictly as a JSON object with the following keys:
    - `technicalScore`: (integer 0-100)
    - `communicationScore`: (integer 0-100)
    - `relevanceScore`: (integer 0-100)
    - `feedback`: (string)

    Example JSON output:
    ```json
    {{
      "technicalScore": 85,
      "communicationScore": 90,
      "relevanceScore": 88,
      "feedback": "The response was technically sound and clearly communicated, showing good understanding."
    }}
    ```

    Question: "{question_obj['question']}"
    Candidate's Response: "{response_text}"
    """
    
    try:
        gemini_response_text = generate_content_with_gemini(response_evaluator_model, prompt)
        
        if gemini_response_text:
            print(f"DEBUG: Raw Gemini response for submit_answer: {gemini_response_text}")
            cleaned_response = extract_json_from_gemini_response(gemini_response_text, is_array=False)
            
            if not cleaned_response:
                raise ValueError("Failed to extract valid JSON from Gemini response for answer evaluation.")
            
            evaluation = json.loads(cleaned_response)
            
            overall_q_score = (evaluation.get('technicalScore', 0) + evaluation.get('communicationScore', 0) + evaluation.get('relevanceScore', 0)) / 3
            evaluation['score'] = round(overall_q_score)
            
            session['interview_responses'].append({
                'question_id': question_id,
                'question': question_obj['question'],
                'tags': question_obj['tags'],
                'response': response_text,
                'duration': duration,
                'evaluation': evaluation
            })
            
            return jsonify({'message': 'Answer submitted and evaluated', 'evaluation': evaluation}), 200
        else:
            return jsonify({'error': 'AI failed to evaluate response or returned empty response.'}), 500

    except json.JSONDecodeError as e:
        print(f"DEBUG: JSON Decode Error in /submit_answer: {e}")
        print(f"DEBUG: Raw AI response that caused error: {gemini_response_text}")
        return jsonify({'error': f'Failed to parse AI response as JSON for answer evaluation: {e}. Raw AI response: {gemini_response_text}'}), 500
    except ValueError as e:
        print(f"DEBUG: Value Error in /submit_answer: {e}")
        return jsonify({'error': f'{str(e)}. Raw AI response: {gemini_response_text}'}), 500
    except Exception as e:
        print(f"DEBUG: Error during response evaluation in /submit_answer: {e}")
        return jsonify({'error': f'An unexpected error occurred during AI processing: {str(e)}'}), 500

@app.route('/get_assessment', methods=['GET'])
def get_assessment():
    session_id = request.headers.get('X-User-Session-Id')
    if not session_id or session_id not in sessions:
        return jsonify({'error': 'Invalid or missing session ID'}), 400
    session = sessions[session_id]

    if not session['interview_responses']:
        return jsonify({'error': 'No interview responses to assess'}), 400

    session['interview_end_time'] = datetime.now().isoformat()

    candidate_profile = session['candidate_profile']
    interview_summary = []
    total_duration_seconds = 0

    for res in session['interview_responses']:
        interview_summary.append(f"Q: {res['question']}\nA: {res['response']}\nEvaluation: Technical: {res['evaluation']['technicalScore']}%, Communication: {res['evaluation']['communicationScore']}%, Relevance: {res['evaluation']['relevanceScore']}%. Feedback: {res['evaluation']['feedback']}")
        
        try:
            minutes, seconds = map(int, res['duration'].split(':'))
            total_duration_seconds += (minutes * 60) + seconds
        except ValueError:
            pass

    total_minutes = total_duration_seconds // 60
    total_remaining_seconds = total_duration_seconds % 60
    interview_duration_str = f"{total_minutes}m {total_remaining_seconds}s"


    prompt = f"""Generate a comprehensive interview assessment report based on the following candidate profile and interview responses.
    
    Candidate Profile: {json.dumps(candidate_profile, indent=2)}
    
    Interview Questions and Responses:
    {'-'*30}
    {"\n\n".join(interview_summary)}
    {'-'*30}

    Overall Interview Duration: {interview_duration_str}

    Provide the assessment strictly as a JSON object with the following structure:
    - `overallScore`: (integer 0-100, aggregate score based on all responses)
    - `recommendation`: (string, e.g., "Highly Recommended", "Recommended", "Consider with Reservations", "Not Recommended")
    - `interviewDuration`: (string, e.g., "15m 30s")
    - `detailedScores`: (object with `technicalSkills`, `communication`, `softSkills` - each an integer 0-100)
    - `detailedQuestionAnalysis`: (array of objects, one for each question, including `question`, `response`, `tags`, `score`, `technicalScore`, `communicationScore`, `relevanceScore`)
    - `keyStrengths`: (array of strings)
    - `areasForImprovement`: (array of strings)

    Example JSON output:
    ```json
    {{
      "overallScore": 85,
      "recommendation": "Recommended",
      "interviewDuration": "12m 45s",
      "detailedScores": {{
        "technicalSkills": 88,
        "communication": 82,
        "softSkills": 85
      }},
      "detailedQuestionAnalysis": [
        {{
          "question": "Tell me about a challenging project...",
          "response": "My response...",
          "tags": ["experience"],
          "score": 80,
          "technicalScore": 75,
          "communicationScore": 85,
          "relevanceScore": 80
        }}
      ],
      "keyStrengths": ["Strong technical foundation"],
      "areasForImprovement": ["More detailed examples"]
    }}
    ```
    """

    try:
        gemini_response_text = generate_content_with_gemini(assessment_generator_model, prompt)

        if gemini_response_text:
            print(f"DEBUG: Raw Gemini response for assessment_generator: {gemini_response_text}")
            cleaned_response = extract_json_from_gemini_response(gemini_response_text, is_array=False)

            if not cleaned_response:
                raise ValueError("Failed to extract valid JSON from Gemini response for assessment generation.")

            assessment = json.loads(cleaned_response)
            assessment['interviewDuration'] = interview_duration_str
            session['interview_assessment'] = assessment
            return jsonify({'message': 'Assessment generated', 'assessment': assessment}), 200
        else:
            return jsonify({'error': 'AI failed to generate assessment or returned empty response.'}), 500

    except json.JSONDecodeError as e:
        print(f"DEBUG: JSON Decode Error in /get_assessment: {e}")
        print(f"DEBUG: Raw AI response that caused error: {gemini_response_text}")
        return jsonify({'error': f'Failed to parse AI response as JSON for assessment generation: {e}. Raw AI response: {gemini_response_text}'}), 500
    except ValueError as e:
        print(f"DEBUG: Value Error in /get_assessment: {e}")
        return jsonify({'error': f'{str(e)}. Raw AI response: {gemini_response_text}'}), 500
    except Exception as e:
        print(f"DEBUG: Error during assessment generation in /get_assessment: {e}")
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500


if __name__ == '__main__':
    port = 5000 
    
    print(f"Flask app running on http://127.0.0.1:{port}")
    app.run(host='127.0.0.1', port=port, debug=True)

