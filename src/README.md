# Ideal Least Squares Function Finder 

This is a script designed to identify the top four functions that best match a given training function using the Least Squares method in a computationally efficient way.

## Running the Docker Container

To build and run the Docker container, follow these steps:

1. Open Docker Desktop.
2. In your terminal, enter:
   ```sh
   docker build -t ideal-finder-app .
   ```
3. To run the website, enter: 

```sh 
docker run -p 4000:80 ideal-finder-app
```

## Running Unit Tests 
To run the unit tests, ensure you have **pytest** installed (ideally within a virtual environment). Then, you can run the tests by entering this in your terminal (not in your Python shell!):

```sh
python -m unittest discover -s tests -v
```

## Addiitonal Notes 
### Additional Task Notebook 
In additional_**task.ipynb**, you will find a step-by-step explanation of how the code can be pushed to a remote repository on GitHub.

Code Explanation
Check the Jupyter notebook for a step-by-step explanation of how the code works.
---
David Valls Lanaquera
