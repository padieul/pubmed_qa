# Use an official Python runtime as a base image
FROM python:3.8

# Set the working directory to /app/middleware
WORKDIR /app/middleware

# Copy the requirements.txt file to the working directory
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt
# RUN pip install huggingface_hub
# RUN pip install --upgrade sentence_transformers huggingface_hub

# Copy the rest of the application code to the working directory
COPY . .

# Expose port 8000 (assuming your FastAPI application is running on this port)
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
