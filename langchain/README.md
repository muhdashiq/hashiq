#### Deprecation warnings about Langchain 0.1.0 and 0.2.0 should be ignored as we are not using these versions!

## Python Version
First, you must have the 3.11 version of Python installed:

https://www.python.org/downloads/

This is very important, as only a few versions of Python support LangChain and OpenAI.

## Pipenv Installation and Configuration
1. If you have not already done so, create a tchat directory somewhere on your development machine.
2. If you have not already installed Pipenv from the previous section, you must do that now. In your terminal run pip install pipenv or depending on your environment, pip3 install pipenv

3. Create a file in your tchat project directory called Pipfile

4. Copy paste the following code into that new Pipfile (or drag and drop the file that is attached to this lecture into your tchat project directory):

[[source]]
url = "https://pypi.org/simple"
verify_ssl = true
name = "pypi"
 
[packages]
langchain = "==0.0.352"
openai = "==0.27.8"
python-dotenv = "==1.0.0"
 
[dev-packages]
 
[requires]
python_version = "3.11"


5. Inside your tchat project directory, run the following command to install your dependencies from the Pipfile:

pipenv install

6. Run the following command to create and enter a new environment:

pipenv shell

After doing this your terminal will now be running commands in this new environment managed by Pipenv.

Once inside this shell, you can run Python commands just as shown in the lecture videos.

eg:

python main.py

7. If you make any changes to your environment variables or keys, you may find that you need to exit the shell and re-enter using the pipenv shell command.

*Important* - Anaconda users may find that Pipenv conflicts with their environment. Please deactivate your conda environment if you find this to be true.

*Deprecation warnings about Langchain 0.1.0 and 0.2.0 should be ignored as we are not using these versions!*