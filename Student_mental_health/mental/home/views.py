from django.shortcuts import render,redirect
from django.shortcuts import HttpResponse
from django.contrib.auth.models import User
from django.contrib.auth import authenticate,login
from django.contrib import messages
from django.core.exceptions import ValidationError
import numpy as np
import joblib
from django.contrib.auth import authenticate, login as auth_login
from django.http import JsonResponse
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import re
from collections import defaultdict


# Initialize the LLM with the "llama3" model
model1 = OllamaLLM(model="llama3.2")

# Define the chatbot prompt
template = """
You are a comprehensive Health Support Bot designed to provide accurate, empathetic, and detailed answers 
to health-related questions. You can answer questions on a wide range of topics, including but not limited to:
physical health, mental health, diseases, treatments, symptoms, prevention, nutrition, fitness, wellness, 
chronic illnesses, medications, diagnostics, and therapies.

Always prioritize user safety and provide guidance with disclaimers when medical intervention is required.

Here is the conversation history:
{context}

User's Question: {question}

Your Response:
"""
prompt = ChatPromptTemplate.from_template(template)

# Predefined greetings
greetings = {
    "hello": "Hello! How can I assist you with your health-related queries today?",
    "hi": "Hi there! Feel free to ask me any health-related questions.",
    "good morning": "Good morning! How can I support your health needs today?",
    "good afternoon": "Good afternoon! How can I help with your health-related concerns?",
    "good evening": "Good evening! I'm here to answer your health-related queries.",
    "good night": "Good night! If you have any health-related questions, I'm here to help.",
    "hyy": "Hey there! How can I assist you with your health queries today?",
    "hii": "Hi! How can I assist you with health-related questions?",
    "namaste": "Namaste! How can I help with your health concerns today?",
    "hola": "Hola! How can I assist you with health-related queries?",
    "hey": "Hey! Let me know how I can assist with your health concerns.",
    "howdy": "Howdy! What health-related queries can I assist you with today?",
    "sup": "Hey there! Feel free to ask me about health-related topics.",
    "yo": "Yo! How can I assist you today?",
    "wassup": "Hey! What's up? How can I help with your health concerns?",
    "hai": "Hi! How can I assist you today?",
}

# Ensure case-insensitive matching for greetings
greetings_normalized = {key.lower(): value for key, value in greetings.items()}

# Health-related short forms
short_forms = {
    "bp": "blood pressure",
    "bmi": "body mass index",
    "cpr": "cardiopulmonary resuscitation",
    "rbc": "red blood cells",
    "wbc": "white blood cells",
    "ecg": "electrocardiogram",
    "ct": "computed tomography",
    "mri": "magnetic resonance imaging",
    "ldl": "low-density lipoprotein (bad cholesterol)",
    "hdl": "high-density lipoprotein (good cholesterol)",
    "hiv": "human immunodeficiency virus",
    "tb": "tuberculosis",
    "covid": "coronavirus disease",
    "cbc": "complete blood count",
    "ai": "artificial intelligence (in medical applications)",
}

# Cached responses for common FAQs
faq_responses = {
    "What is blood pressure?": (
        "Blood pressure is the force of blood pushing against the walls of your arteries. "
        "It is measured in two numbers: systolic (the higher number, indicating pressure during a heartbeat) "
        "and diastolic (the lower number, indicating pressure between heartbeats)."
    ),
    "What is cancer?": (
        "Cancer is a group of diseases characterized by the uncontrolled growth and spread of abnormal cells. "
        "It can affect almost any part of the body and is caused by genetic mutations, environmental factors, or lifestyle choices."
    ),
    "How can I manage depression?": (
        "Managing depression involves a combination of self-care, therapy, and possibly medication. "
        "Strategies include staying physically active, seeking support from loved ones, practicing mindfulness, "
        "and consulting a mental health professional for tailored advice."
    ),
}

# Function to identify health-related queries
def is_health_query(user_input):
    keywords = [
        "health", "wellness", "nutrition", "fitness", "disease", "symptoms", "treatment", "medicine",
        "blood pressure", "diabetes", "cancer", "depression", "mental health", "surgery", "therapy",
        "exercise", "diet", "cholesterol", "anxiety", "stress", "vaccination", "infection", "virus",
        "injury", "first aid", "rehabilitation", "addiction", "pain", "arthritis", "asthma",
        "osteoporosis", "cardiology", "dermatology", "neurology", "psychiatry", "pediatrics",
        *short_forms.values()  # Include expanded short forms
    ]
    return any(keyword in user_input.lower() for keyword in keywords)

# Main chatbot function
def chatbot(request):
    if request.method == "POST":
        user_input = request.POST.get("user_input", "").strip()
        context = request.session.get("context", "")  # Retrieve chat history from the session

        normalized_input = user_input.lower()

        # Handle greetings
        if normalized_input in greetings_normalized:
            response = greetings_normalized[normalized_input]
        # Expand short forms
        elif normalized_input in short_forms:
            expanded = short_forms[normalized_input]
            response = f"{normalized_input.upper()} stands for {expanded}. Feel free to ask more!"
        # FAQs
        elif user_input in faq_responses:
            response = faq_responses[user_input]
        # Health queries
        elif is_health_query(user_input):
            try:
                formatted_prompt = prompt.format(context=context, question=user_input)
                response = model1.invoke(formatted_prompt)  # Replace with your model invocation logic
            except Exception as e:
                response = "I'm sorry, something went wrong. Please try again later."
        else:
            response = (
                "I'm sorry, I can only assist with health-related guidance. "
                "Feel free to ask about physical health, mental health, diseases, or general wellness."
            )

        # Update the context
        context += f"User: {user_input}\nAI: {response}\n"
        request.session["context"] = context  # Save the updated context in the session

        # Format the response for the UI
        formatted_response = format_response(response)
        messages = [
            {"sender": "User", "text": user_input},
            {"sender": "AI", "text": formatted_response},
        ]
        return render(request, "chatbot.html", {"messages": messages})

    # On GET, show an empty chat page
    context = request.session.get("context", "")
    messages = []
    if context:
        for line in context.splitlines():
            if line.startswith("User:"):
                messages.append({"sender": "User", "text": line[6:]})
            elif line.startswith("AI:"):
                messages.append({"sender": "AI", "text": line[4:]})
    return render(request, "chatbot.html", {"messages": messages})

def format_response(response):
    """
    Formats the chatbot response for better readability with structured lists and paragraphs.
    """
    response = re.sub(r'(\d+[\.\)]\s)', r'<br><strong>\1</strong>', response)  # Highlight numbered points
    response = re.sub(r'([^\n])\n', r'\1<br>', response)  # Convert newline to <br> for paragraphs

    if response.startswith('<br>'):
        response = response[4:]

    return response


# Load your model
model = joblib.load('static/Student_Mental_Health_Predicator')

# Create a mapping for the predicted values
pred_mapping = {
    1: "Healthy",
    2: "Moderate Affected",
    3: "Severely affected"
}

# Create your views here.

def index(request):
    return render(request,'index.html')

# def chatbot(request):
#     return render(request,'chatbot.html')


def about(request):
    return render(request,'about.html')
def prediction(request):
    output = None 

    if request.method == 'POST':
        try:
            # Fetching form values safely and converting them to the required data types
            gender = int(request.POST.get('gender', 0))  # Gender encoded as 0 or 1
            education_level = int(request.POST.get('education_level', 0))  # Education level
            depressed = int(request.POST.get('depressed', 0))  # Feeling depressed
            interest = int(request.POST.get('interest', 0))  # Little interest or pleasure in doing things
            sleep = int(request.POST.get('sleep', 0))  # Are you having proper sleep (min 6hrs)
            energy = int(request.POST.get('energy', 0))  # Feeling tired or having little energy
            meal = int(request.POST.get('meal', 0))  # Taking proper meals or not
            failure = int(request.POST.get('failure', 0))  # Feeling of failure
            concentration = int(request.POST.get('concentration', 0))  # Trouble concentrating
            restless = int(request.POST.get('restless', 0))  # Restless or slowed movements
            self_harm = int(request.POST.get('self_harm', 0))  # Thoughts of harming yourself
            job = int(request.POST.get('job', 0))  # Part-time/full-time job
            family = int(request.POST.get('family', 0))  # Living with family
            study_hours = int(request.POST.get('study_hours', 0))  # Hours spent studying each day
            gadgets = int(request.POST.get('gadgets', 0))  # Hours spent on electronic gadgets
            social_media = int(request.POST.get('social_media', 0))  # Hours spent on social media
            exercise = int(request.POST.get('exercise', 0))  # Weekly physical activity hours
            substance = int(request.POST.get('substance', 0))  # Consumption of substances (alcohol, tobacco, drugs)
            percentage = float(request.POST.get('percentage', 0.0))  # Last year percentage (decimal number)

            
            input_data = np.array([[gender, education_level, depressed, interest, sleep, energy, meal, failure,
                                   concentration, restless, self_harm, job, family, study_hours, gadgets, 
                                   social_media, exercise, substance, percentage]])

            
            pred = model.predict(input_data)

            
            decoded_pred = pred_mapping.get(pred[0], "Unknown")

           
            output = f"Prediction: {decoded_pred}"

        except ValueError as e:
            
            output = f"Error: {str(e)}"

    
    return render(request, 'prediction.html', {'output': output})
def contact(request):
    return render(request,'contact.html')
def login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            auth_login(request, user)  # Use Django's login function
            return redirect('home')  # Redirect to home page or any desired page
        else:
            # Handle invalid login
            return render(request, 'login.html', {'error': 'Invalid credentials'})
    else:
        return render(request, 'login.html')
    

def signup(request):
    if request.method == 'POST':
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        username = request.POST.get('username')
        email = request.POST.get('email_or_phone')  # Adjusted field name to match the form
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirm_password')

        # Check if passwords match
        if password != confirm_password:
            messages.error(request, "Passwords do not match.")
            return render(request, 'signup.html')

        # Check if username is already taken
        if User.objects.filter(username=username).exists():
            messages.error(request, "Username is already taken.")
            return render(request, 'signup.html')

        # Check if username is alphanumeric and not purely numeric
        if not username.isalnum() or username.isnumeric():
            messages.error(request, "Username must contain both letters and numbers, and it can't be purely numeric.")
            return render(request, 'signup.html')

        # Create the user and set additional fields
        try:
            user = User.objects.create_user(username=username, email=email, password=password)
            user.first_name = first_name  # Set first name
            user.last_name = last_name  # Set last name
            user.save()

            messages.success(request, "Account created successfully.")
            return redirect('login')  # Redirect to login page after successful signup
        except ValidationError as e:
            messages.error(request, str(e))
            return render(request, 'signup.html')

    return render(request, 'signup.html')




def profile(request):
    if request.user.is_authenticated:
        return render(request, 'profile.html', {'user': request.user})
    else:
        return redirect('login')  # Redirect to login if not authenticated
