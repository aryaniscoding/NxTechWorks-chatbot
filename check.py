import os
import google.generativeai as genai

# configure with your API key
genai.configure(api_key="AIzaSyCmSJFWTprWbqsyJelQryC1Al1pF_FB8zA")

# this helper was added in newer versions:
models = genai.list_models()

for m in models:
    print(f"Model: {m.name}")
    print("  Supported methods:", m.supported_generation_methods)
    print()
