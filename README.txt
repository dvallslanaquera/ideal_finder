This is the app created for the written assignment for the course 'Programming with Python'
There is a Jupyter notebook that works as the documentation of the projection.
There you can find a step-by-step explanation of how the code works. 

David Valls Lanaquera


-- To run Docker
Open Docker Desktop. Then in your terminal enter: 
docker build -t ideal-finder-app

Then, to run the website you should enter:
docker run -p 4000:80 ideal-finder-app


-- To run unit tests
You need pytest to be installed (ideally while using the virtual environment)
Then, you can run the tests entering this in your terminal (not in your Python shell!):

python -m unittest discover -s tests -v


-- Note: about additional_task.ipynb
In this notebook you will find a step-by-step explanation of how the code can be pushed to a remote repository in GitHub 