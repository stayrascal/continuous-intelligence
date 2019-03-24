#!/usr/bin/env bash

artifact_root_dir=$PWD/artifact
file_store=$PWD/log
mlflow server --file-store $file_store --default-artifact-root $artifact_root_dir