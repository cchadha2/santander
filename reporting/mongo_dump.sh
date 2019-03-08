#!/usr/bin/env bash

# Dump mongo archive
docker exec reporting_mongodb_1 sh -c 'exec mongodump -d reporting --archive' > /Users/cchadha2/Documents/Github-Private/santander/reporting/data/lgb_valid.archive

# Dump archive as csv
docker exec reporting_mongodb_1 sh -c 'mongoexport --db reporting --collection validation --type csv --fieldFile /data/fields.txt' > /Users/cchadha2/Documents/Github-Private/santander/reporting/data/lgb_valid.csv