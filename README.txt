
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


* Check the jupyter notebook for a step-by-step explanation of how the code works.

David Valls Lanaquera
