#!/usr/bin/env bash
# Restore mongo container with dump (inside bash shell in container)
mongorestore --nsFrom reporting.validation --nsTo reporting.validation --archive=/data/lgb_valid.archive