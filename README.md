Project "Jarvis Assistant"

core/: The heart of the application. Each file is responsible for its own part (STT, AI, TTS, Audio).
config/: Stores settings and secrets separately from the code. Never add the .env file to Git!
main.py: Puts everything together and runs the app.
requirements.txt: Makes it easy to install all required libraries on another machine or in a new environment.
.venv/: Folder with the virtual environment. Always added to .gitignore.


model: gemini-2.5-pro-exp-03-25
