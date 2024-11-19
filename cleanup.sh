#!/bin/bash

TARGET_DIR="./Tarefa2/submissions"
TARGET_DIR2="./Tarefa2/graphs"

find "$TARGET_DIR" -type f ! -name '.gitignore' -delete
find "$TARGET_DIR2" -type f ! -name '.gitignore' -delete