#!/usr/bin/env bash

isort . --float-to-top --multi-line=3 --trailing-comma --force-grid-wrap=0 --use-parentheses --line-width=88	
black -l 88 .	
flake8 . --max-complexity=12 --max-line-length=88 --select=C,E,F,W,B,B950,BLK --ignore=E203,E231,E501,W503 --exclude=.cache
