# Use an official Python runtime as the base image
FROM heroku/heroku:20

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the command to run your app using gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:$PORT"]