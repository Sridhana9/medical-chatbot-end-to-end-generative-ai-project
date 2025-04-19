# medical-chatbot-end-to-end-generative-ai-project

Introduction

This repository contains the source code for a Medical Chatbot that leverages Mistral AI and Pinecone for intelligent responses. The chatbot is built using Python, LangChain, Flask, and deployed on AWS using EC2 and ECR.

Tech Stack Used

Python

LangChain

Flask

Mistral AI

Pinecone

AWS (EC2, ECR, GitHub Actions)

Installation and Setup

Step 1: Clone the Repository

git clone https://github.com/YOUR_REPO.git cd YOUR_REPO

Step 2: Create a Conda Environment

conda create -n medibot python=3.10 -y conda activate medibot

Step 3: Install Dependencies

pip install -r requirements.txt

Step 4: Set Up Environment Variables

Create a .env file in the root directory and add your Pinecone and OpenAI credentials:

PINECONE_API_KEY="your_pinecone_api_key" OPENAI_API_KEY="your_openai_api_key"

Step 5: Store Embeddings to Pinecone

python store_index.py

Step 6: Run the Application

python app.py

Step 7: Access the Chatbot

Open your browser and navigate to:

http://localhost:8080
