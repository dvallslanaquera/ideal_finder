FROM python:3.9.6
WORKDIR /usr/src/app
COPY . .
# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# Make port 80 available
EXPOSE 80
# Run app.py when the container launches
CMD ["python", "./main.py"]
