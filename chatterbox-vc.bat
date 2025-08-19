@echo off

echo --- Environment Diagnostics ---
echo.
echo --- Python and Pip Location ---
where python
where pip
echo.
echo --- Pip Version ---
python -m pip --version
echo.
echo --- Installed Packages ---
python -m pip list
echo.
echo --- End of Diagnostics ---
echo.

REM This script automates the setup of the Chatterbox project.
REM It assumes you have Python and Git installed on your system.

echo --- Chatterbox Project Setup ---

REM Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in your PATH.
    echo Please install Python and try again.
    pause
    exit /b
)

REM Check for Git
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Git is not installed or not in your PATH.
    echo Please install Git and try again.
    pause
    exit /b
)

echo.
echo Step 1: Cloning the project repository from GitHub...
git clone https://github.com/heavensarder/chaterbox-vc.git

REM Check if the clone was successful
if not exist chaterbox-vc (
    echo.
    echo Error: Failed to clone the repository.
    echo Please check your internet connection.
    pause
    exit /b
)

cd chaterbox-vc

echo.
echo Step 2: Creating requirements.txt file...

(echo numpy>=1.26.0
echo librosa==0.11.0
echo s3tokenizer
echo torch==2.6.0
echo torchaudio==2.6.0
echo transformers==4.46.3
echo diffusers==0.29.0
echo resemble-perth==1.0.1
echo conformer==0.3.2
echo safetensors==0.5.3
echo gradio) > requirements.txt

echo.
echo Step 3: Installing Python libraries from requirements.txt...
python -m pip install -r requirements.txt

echo.
echo Step 3.5: Installing project in editable mode...
python -m pip install -e .

if %errorlevel% neq 0 (
    echo.
    echo Error: Failed to install Python libraries.
    echo Please check your internet connection and pip installation.
    pause
    exit /b
)

echo.
echo Step 4: Starting the application...
echo The first time the application runs, it will download the model files.
echo This may take a few minutes depending on your internet speed.
echo.
python app.py

echo.
echo --- Setup Complete ---
pause