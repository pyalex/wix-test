VIRTUAL_ENV	?= venv
FLASK_APP ?= app.py


$(VIRTUAL_ENV): $(CURDIR)/requirements.txt
	@[ -d $(VIRTUAL_ENV) ] || python3 -m venv $(VIRTUAL_ENV)
	@$(VIRTUAL_ENV)/bin/pip install -r requirements.txt
	@touch $(VIRTUAL_ENV)

$(VIRTUAL_ENV)/bin/py.test: $(VIRTUAL_ENV) $(CURDIR)/requirements-tests.txt
	@$(VIRTUAL_ENV)/bin/pip install -r requirements-tests.txt
	@touch $(VIRTUAL_ENV)/bin/py.test

test: $(VIRTUAL_ENV)/bin/py.test
	@$(VIRTUAL_ENV)/bin/py.test $(CURDIR)/tests.py

run: $(VIRTUAL_ENV) 
	FLASK_APP=$(FLASK_APP) $(VIRTUAL_ENV)/bin/python -m flask run
