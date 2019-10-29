
report:
	allure serve reports/allure

pytest:
	pipenv run pytest --cov --cov-fail-under=9 --html=reports/report.html --alluredir=reports/allure

check:
	pipenv run flake8

black:
	pipenv run black

train:
	pipenv run python src/decision_tree.py

run:
	pipenv run python src/app.py
