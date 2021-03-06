#!/bin/bash

find . -type f -size +10M > .gitignore
echo '*.csv' >> .gitignore
echo '*.xlsx' >> .gitignore
echo '*.pkl' >> .gitignore
echo '*.json' >> .gitignore
echo './._*' >> .gitignore

