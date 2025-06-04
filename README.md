# Installation

In your terminal navigate to the folder you want to clone this repo into and clone with 
```
git clone https://github.com/Landscape-CV/PyTLiDAR.git
cd PyTLiDAR
```
## Create a .venv & requirements installed
### Mac
```
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```
### Windows
```
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt

```
## Create a .env file
make a file in your parent director named `.env` and paste the following into it replace the ... with the filepath to your data
```
DATA_FILE_PATH = ...
```