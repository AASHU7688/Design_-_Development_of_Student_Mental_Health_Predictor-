# Design_and_Development_of_Student_Mental_Health_Predictor-
This project aims to address the growing concern of mental health among students by providing a machine learning-based web application that predicts the mental health condition of students based on various parameters. The application is designed to offer insights that can assist students, parents, and professionals in taking proactive steps towards mental wellness.

# Features
Custom Dataset: The dataset was created using a Google Form survey distributed in colleges and schools. It includes a variety of features relevant to mental health.

Machine Learning Model: A logistic regression multiclass classifier is used to predict mental health conditions categorized into different levels.

User-Friendly Interface:
Frontend developed using HTML, CSS, and JavaScript for an interactive and responsive user experience.

Backend built with Django for robust and secure data handling.

SQLite Database: Stores user data efficiently for seamless integration with the web application.

Chatbot Integration: A chatbot is incorporated to answer questions related to mental health, providing guidance and resources to users.

Data Visualization: Key insights from the dataset are presented in a user-friendly manner for better understanding.
# How It Works

Users input their details through an intuitive form on the web application.

The data is processed and passed to the trained logistic regression model for predictions.

The model predicts the user's mental health condition and displays it in an understandable format.

Users can interact with the integrated chatbot for further queries and support.



# Technology Stack

Frontend: HTML, CSS, JavaScript

Backend: Django

Database: SQLite

Machine Learning: Logistic Regression (Multiclass Classifier)

# Requirement for the project
Django Framework:

Version: 4.x or higher (adjust based on the project).
Reason: Leverages modern Django features like class-based views and enhanced ORM capabilities.

Python:

Version: 3.10 or higher.
Reason: Compatible with modern libraries, type hints, and async support.

# Libraries:

NumPy: >=1.21.0
For numerical operations in prediction functions.

joblib: >=1.2.0
For loading serialized machine learning models.

LangChain Ollama: >=0.1.0
Handles interaction with the Ollama LLM.

LangChain Core Prompts: >=0.1.0
To manage prompt templates for the LLM.

Django Session Middleware: Ensure it is enabled for session-based context storage.

Ollama Version:

Version: 3.2 (model-specific: "llama3.2").

Reason: Supports advanced health-related queries with a large knowledge base.

Front-End Libraries:

HTML/CSS for templates.

Optional: Bootstrap 5.x for responsive design.

Database:

SQLite (default for development) or PostgreSQL for production.

Deployment Requirements:

Gunicorn: >=20.0.0
WhiteNoise: >=5.3.0 (for serving static files in production).

# Steps or command to run the project 

1) Download the zip and extract the zip.
2) Open the extracted folder on the VSCode
3) Open terminal of the VSCode.
4) Hit this command first on the terminal - cd mental
5) Then next another command on the terminal after 3rd step - python manage.py runserver
