# Prosthetics Agent
Capstone project

Ben Lambright

## Features
- Front-end chatbot interface
- Integrating a LaTeX and planning tool to improve outputs for solving math problems and generating formulas
- Dockerized for ease of use

## Model Architecture
See our report for a more detailed explanation, but here is the general architecture of our model
![architecture](images/User_Input.png)

## Repo Structure
```text
math-tutor/
├── agent.py               # model
├── app.py                 # runs the Flask app
├── agent_test.py          # Example test script for the agent
├── conversation_test.py   # Another example test for conversation flow
├── images/
│   └── User_Input.png     
├── templates/
│   └── index.html         
├── Dockerfile             
├── README.md              
├── requirements.txt       
├── secrets.txt            # API keys (do NOT commit publicly)
├── tools.py               # toolkits for model
└── utils.py               
```


## Use 
Obtain all of the API keys necessary, as shown in `secrets.txt`. You will need to go to each of those websites and make an account to get those keys. When adding them to the .txt file, put a space between the name of the API and the API key. The following is an example for within secrets.txt.
```text
API_KEY key_image_code
```

Clone the repo, and cd to the directory. Once you are cd'ed into the directory, you can run the following commands.

Build the docker app
```shell
# Build the image
docker build -t math_tutor .
```
Once you have run the Docker app, make sure to run the image as so, in order for you to be able to access the port:
```shell
# Run the image
docker run -p [PORT]:5000 math_tutor
```
Finally, you should be able to access the website on the following page: `localhost:[PORT]`
