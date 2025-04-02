python --version

pip install --upgrade pip

echo Create python virtual environment
python3.10 -m venv env

echo Activate virtual environment and install dependencies
call env/Scripts/activate

pip install -r requirements.txt

code .