#!/bin/bash
IFS='
'
for filename in $(ls ../../../../data/aolivepe/test)
do
	identify -format "%n\n" ../../../../data/aolivepe/test/$filename | head -1
done